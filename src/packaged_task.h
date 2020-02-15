#ifndef PACKAGED_TASK_H
#define PACKAGED_TASK_H

#include "iterator.h"
#include "celerity_helper.h"

#include "computation_type.h"

#include "packaged_task_traits.h"

#include "packaged_tasks/packaged_transform.h"
#include "packaged_tasks/packaged_generate.h"
#include "packaged_tasks/packaged_zip.h"

#include "computation_type_traits.h"
#include "fusion.h"

namespace celerity::algorithm
{

template <typename T>
inline constexpr bool is_source_v = detail::computation_type_of_v<T, computation_type::generate>;

template <typename T>
inline constexpr bool single_element_access_v = (detail::computation_type_of_v<T, computation_type::transform> &&
                                                 detail::access_type_v<T> == access_type::one_to_one) ||
                                                (detail::computation_type_of_v<T, computation_type::zip> &&
                                                 detail::access_type_v<T> == access_type::one_to_one);

template <typename T>
inline constexpr bool is_linkable_source_v = detail::is_partially_packaged_task_v<T> &&
                                                 detail::stage_requirement_v<T> == stage_requirement::output;

template <typename T>
inline constexpr bool is_linkable_sink_v = detail::is_partially_packaged_task_v<T> &&
                                           detail::stage_requirement_v<T> == stage_requirement::input;

template <typename T>
inline constexpr bool is_transiently_linkable_source_v = is_linkable_source_v<T>; // && (is_source_v<T> || single_element_access_v<T>);

template <typename T>
inline constexpr bool is_transiently_linkable_sink_v = is_linkable_sink_v<T> && single_element_access_v<T>;

template <typename T>
inline constexpr bool has_transient_output_v = is_transient_v<typename detail::packaged_task_traits<T>::output_iterator_type>;

template <typename T>
inline constexpr bool has_transient_input_v = is_transient_v<typename detail::packaged_task_traits<T>::input_iterator_type>;

template <typename T>
inline constexpr bool is_fusable_source_v = detail::is_packaged_task_v<T> && has_transient_output_v<T>;

template <typename T>
inline constexpr bool is_fusable_sink_v = detail::is_packaged_task_v<T> && has_transient_input_v<T>;

template <typename T, typename U, std::enable_if_t<is_transiently_linkable_source_v<T> && 
                                                   is_transiently_linkable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    using value_type = typename detail::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = detail::packaged_task_traits<T>::rank;

    transient_buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U, std::enable_if_t<is_linkable_source_v<T> && is_linkable_sink_v<U> && 
                                                   (!is_transiently_linkable_source_v<T> || !is_transiently_linkable_sink_v<U>), int> = 0>
auto operator|(T lhs, U rhs)
{
    using value_type = typename detail::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = detail::packaged_task_traits<T>::rank;

    buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U, std::enable_if_t<is_sequence_v<T> && is_linkable_source_v<last_element_t<T>> && is_linkable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return remove_last_element(lhs) | (get_last_element(lhs) | rhs);
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && detail::is_partially_packaged_task_v<U> && detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    auto t_right = rhs.complete(lhs.get_out_iterator(), lhs.get_out_iterator());

    return sequence(lhs, t_right);
}

template <typename T, typename U, std::enable_if_t<is_sequence_v<T> && 
                                                   detail::is_packaged_task_v<last_element_t<T>> && 
                                                   detail::is_partially_packaged_task_v<U> && 
                                                   detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    return remove_last_element(lhs) | (get_last_element(lhs) | rhs);
}

template <typename T, typename U, std::enable_if_t<is_fusable_source_v<T> && 
                                                   detail::computation_type_of_v<T, computation_type::transform> && 
                                                   is_fusable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return sequence(package_transform<access_type::one_to_one, true>(fuse(lhs.get_task(), rhs.get_task()),
                                                                     lhs.get_in_beg(),
                                                                     lhs.get_in_end(),
                                                                     rhs.get_out_iterator()));

    // Results in a linker error. Not sure why -> need further clarification from philip/peter
    //
    // return package_transform<access_type::one_to_one, true>(task<new_execution_policy>(seq),
    //                                                     lhs.get_in_beg(),
    //                                                     lhs.get_in_end(),
    //                                                     t.get_out_iterator());
}

template <typename T, typename U, std::enable_if_t<is_fusable_source_v<T> && 
                                                   detail::computation_type_of_v<T, computation_type::generate> && 
                                                   is_fusable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    using output_value_type = typename detail::packaged_task_traits<U>::output_value_type;

    auto out_beg = rhs.get_out_iterator();
    auto out_end = end(out_beg.get_buffer());

    return sequence(package_generate<output_value_type, true>(fuse(lhs.get_task(), rhs.get_task()), out_beg, out_end));
}

template <typename T, typename U, std::enable_if_t<is_fusable_source_v<T> && 
                                                   detail::computation_type_of_v<T, computation_type::zip> && 
                                                   is_fusable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    constexpr auto first_input_access_type = detail::packaged_task_traits<U>::access_type;
    constexpr auto second_input_access_type = detail::extended_packaged_task_traits<U, computation_type::zip>::second_access_type;

    return sequence(package_zip<first_input_access_type, second_input_access_type, true>(fuse(lhs.get_task(), rhs.get_task()), 
        lhs.get_in_beg(),
        lhs.get_in_end(),
        lhs.get_second_in_beg(),
        rhs.get_out_iterator()));
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_sequence_v<T> && 
                                                   is_fusable_source_v<last_element_t<T>> && 
                                                   is_fusable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return remove_last_element(lhs) | (get_last_element(lhs) | rhs);
}

template <typename... Actions, size_t... Is>
auto fuse(const sequence<Actions...> &s, std::index_sequence<Is...>)
{
    const auto &actions = s.actions();
    return (... | (std::get<Is>(actions)));
}

template <typename... Actions>
auto fuse(const sequence<Actions...> &s)
{
    return fuse(s, std::make_index_sequence<sizeof...(Actions)>{});
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T> || detail::is_packaged_task_sequence_v<T>, int> = 0>
auto operator|(T lhs, distr_queue q)
{
    if constexpr (detail::is_packaged_task_v<T> || (detail::is_packaged_task_sequence_v<T> && size_v<T> == 1))
    {
        return std::invoke(lhs, q);
    }
    else
    {
        auto r = std::invoke(lhs, q);
        return std::get<std::tuple_size_v<decltype(r)> - 1>(r);
    }
}

template <typename T, std::enable_if_t<is_sequence_v<T> &&
                                           detail::is_partially_packaged_task_v<last_element_t<T>> &&
                                           detail::stage_requirement_v<last_element_t<T>> == stage_requirement::output,
                                       int> = 0>
auto terminate(T seq)
{
    using last_element_type = last_element_t<T>;
    using traits = detail::packaged_task_traits<last_element_type>;

    using value_type = typename traits::output_value_type;
    constexpr auto rank = traits::rank;

    auto last = get_last_element(seq);

    buffer<value_type, rank> out_buf{last.get_range()};

    return append(remove_last_element(seq), last.complete(begin(out_buf), end(out_buf)));
}

template <typename T, std::enable_if_t<detail::is_partially_packaged_task_v<T> &&
                                           detail::stage_requirement_v<T> == stage_requirement::output,
                                       int> = 0>
auto operator|(T lhs, distr_queue q)
{
    return terminate(sequence(lhs)) | q;
}

template <typename T, std::enable_if_t<is_sequence_v<T> &&
                                           detail::is_partially_packaged_task_v<last_element_t<T>> &&
                                           detail::stage_requirement_v<last_element_t<T>> == stage_requirement::output,
                                       int> = 0>
auto operator|(T lhs, distr_queue q)
{
    return fuse(terminate(lhs)) | q;
}

} // namespace celerity::algorithm

namespace celerity
{
template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::is_linkable_sink_v<U>, int> = 0>
auto operator|(celerity::buffer<T, Rank> &lhs, U rhs)
{
    return rhs.complete(begin(lhs), end(lhs));
}

template <typename T, int Rank, typename U, 
          std::enable_if_t<algorithm::is_linkable_sink_v<U>, int> = 0>
auto operator<<(U lhs, celerity::buffer<T, Rank> &rhs)
{
    return lhs.complete(begin(rhs), end(rhs));
}

} // namespace celerity

#endif // PACKAGED_TASK_H