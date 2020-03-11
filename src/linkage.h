#ifndef LINKAGE_H
#define LINKAGE_H

#include "linkage_traits.h"

#include "transient.h"

namespace celerity::algorithm
{

template <typename T, typename U>
auto link_transiently(T lhs, U rhs)
{
    using value_type = typename detail::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = detail::packaged_task_traits<T>::rank;

    transient_buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U>
auto link(T lhs, U rhs)
{
    using value_type = typename detail::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = detail::packaged_task_traits<T>::rank;

    buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U, std::enable_if_t<is_linkable_source_v<T> && is_linkable_sink_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    if constexpr (is_transiently_linkable_source_v<T> &&
                  is_transiently_linkable_sink_v<U>)
    {
        return link_transiently(lhs, rhs);
    }
    else
    {
        return link(lhs, rhs);
    }
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

template <typename T, typename U, std::enable_if_t<is_sequence_v<T> && detail::is_packaged_task_v<last_element_t<T>> && detail::is_partially_packaged_task_v<U> && detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    return remove_last_element(lhs) | (get_last_element(lhs) | rhs);
}

template <typename T, typename U, std::enable_if_t<is_linkable_source_v<last_element_t<U>> && is_sequence_v<U> && is_linkable_sink_v<T>, int> = 0>
auto operator<<(T lhs, U rhs)
{
    auto linked_sequence = link(get_last_element(rhs), lhs);

    auto linked_lhs = get_last_element(linked_sequence);
    auto linked_last_rhs = get_first_element(linked_sequence);

    return partial_t_joint{linked_lhs, append(remove_last_element(rhs), linked_last_rhs)};
}

template <typename T, typename U, std::enable_if_t<is_sequence_v<U> && is_linkable_source_v<last_element_t<U>> && is_sequence_v<T> && is_linkable_sink_v<last_element_t<T>>, int> = 0>
auto operator<<(T lhs, U rhs)
{
    auto linked_sequence = link(get_last_element(rhs), get_last_element(lhs));

    auto linked_last_lhs = get_last_element(linked_sequence);
    auto linked_last_rhs = get_first_element(linked_sequence);

    partial_t_joint joint{linked_last_lhs, append(remove_last_element(rhs), linked_last_rhs)};

    return append(remove_last_element(lhs), joint);
}

template <typename T, typename U, std::enable_if_t<algorithm::is_linkable_sink_v<T> && algorithm::is_linkable_source_v<U>, int> = 0>
auto operator<<(T lhs, U rhs)
{
    return lhs << sequence(rhs);
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

} // namespace celerity::algorithm

namespace celerity
{
template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::is_linkable_sink_v<U>, int> = 0>
auto operator|(celerity::buffer<T, Rank> lhs, U rhs)
{
    return rhs.complete(begin(lhs), end(lhs));
}

template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::is_linkable_sink_v<T>, int> = 0>
auto operator<<(T lhs, celerity::buffer<U, Rank> rhs)
{
    return lhs.complete(begin(rhs), end(rhs));
}

template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::is_linkable_source_v<T> && algorithm::detail::partially_packaged_task_traits<T>::requirement == algorithm::stage_requirement::output, int> = 0>
auto operator|(T lhs, celerity::buffer<U, Rank> rhs)
{
    return lhs.complete(begin(rhs), end(rhs));
}

template <typename T, int Rank, typename U,
          std::enable_if_t<algorithm::is_sequence_v<T> &&
                               algorithm::is_linkable_source_v<algorithm::last_element_t<T>> &&
                               algorithm::detail::partially_packaged_task_traits<algorithm::last_element_t<T>>::requirement == algorithm::stage_requirement::output,
                           int> = 0>
auto operator|(T lhs, celerity::buffer<U, Rank> rhs)
{
    return remove_last_element(lhs) | (get_last_element(lhs) | rhs);
}

} // namespace celerity

#endif // LINKAGE_H