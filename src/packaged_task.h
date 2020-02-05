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

template <typename T, typename U, std::enable_if_t<detail::is_partially_packaged_task_v<T> && 
                                                   detail::stage_requirement_v<T> == stage_requirement::output && 
                                                   detail::is_partially_packaged_task_v<U> && 
                                                   detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    using value_type = typename T::output_value_type;

    transient_buffer<value_type, T::rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U, std::enable_if_t<is_sequence_v<T> && 
                                                   detail::is_partially_packaged_task_v<last_element_t<T>> &&
                                                   detail::stage_requirement_v<last_element_t<T>> == stage_requirement::output && 
                                                   detail::is_partially_packaged_task_v<U> && 
                                                   detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    return lhs | (get_last_element(lhs) | rhs);
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && 
                                                   detail::is_partially_packaged_task_v<U> && 
                                                   detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    auto t_right = rhs.complete(lhs.get_out_iterator(), lhs.get_out_iterator());

    return sequence(lhs, t_right);
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && 
                                                   detail::computation_type_of_v<T, computation_type::transform> &&
                                                   detail::is_packaged_task_v<U>  && 
                                                   detail::computation_type_of_v<U, computation_type::transform>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return package_transform<access_type::one_to_one, true>(fuse(lhs.get_task(), rhs.get_task()),
                                                            lhs.get_in_beg(),
                                                            lhs.get_in_end(),
                                                            rhs.get_out_iterator());

    // Results in a linker error. Not sure why -> need further clarification from philip/peter
    //
    // return package_transform<access_type::one_to_one, true>(task<new_execution_policy>(seq),
    //                                                     lhs.get_in_beg(),
    //                                                     lhs.get_in_end(),
    //                                                     t.get_out_iterator());
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && 
                                                   detail::computation_type_of_v<T, computation_type::generate> &&
                                                   detail::is_packaged_task_v<U> && 
                                                   detail::computation_type_of_v<U, computation_type::transform>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return package_generate<access_type::one_to_one, true>(fuse(lhs.get_task(), rhs.get_task()), rhs.get_out_beg(), rhs.get_out_end());                                           
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && detail::is_packaged_task_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return sequence(lhs, rhs);
}

template<typename...Actions, size_t...Is>
auto fuse(const sequence<Actions...>& s, std::index_sequence<Is...>)
{
    const auto& actions = s.get_actions();
    return sequence( (... | (std::get<Is>(s))) );
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T> || detail::is_packaged_task_sequence_v<T>, int> = 0>
auto operator|(T lhs, distr_queue q)
{
    fuse(lhs, std::make_index_sequence<size_v<T> - 1>{});

    if constexpr (detail::is_packaged_task_v<T>)
        return std::invoke(lhs, q);
    else
        return std::get<size_v<T> - 1>(std::invoke(lhs, q));
}

template <typename T, std::enable_if_t<detail::is_partially_packaged_task_v<T> &&
                                       detail::stage_requirement_v<T> == stage_requirement::output, int> = 0>
auto operator|(T lhs, distr_queue q)
{
    using value_type = typename T::output_value_type;

    buffer<value_type, T::rank> out_buf{lhs.get_range()};
    
    auto t = lhs.complete(begin(out_buf), end(out_buf));

    return sequence(lhs, q);
}

template <typename T, std::enable_if_t<is_sequence_v<T> && 
                                       detail::is_partially_packaged_task_v<last_element_t<T>> &&
                                       detail::stage_requirement_v<last_element_t<T>> == stage_requirement::output, int> = 0>
auto operator|(T lhs, distr_queue q)
{
    return lhs | (get_last_element(lhs) | q);
}

} // namespace celerity::algorithm

namespace celerity
{
template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::detail::is_partially_packaged_task_v<U> && algorithm::detail::stage_requirement_v<U> == algorithm::stage_requirement::input, int> = 0>
auto operator|(celerity::buffer<T, Rank> &lhs, U rhs)
{
    return rhs.complete(begin(lhs), end(lhs));
}

// template <typename T, int Rank, typename U, std::enable_if_t<algorithm::detail::is_placeholder_task_v<U, algorithm::buffer_iterator<T, Rank>>, int> = 0>
// auto operator<<(U lhs, celerity::buffer<T, Rank> &rhs)
// {
//     return lhs(begin(rhs), end(rhs));
// }

} // namespace celerity

#endif // PACKAGED_TASK_H