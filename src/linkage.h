#ifndef LINKAGE_H
#define LINKAGE_H

#include "linkage_traits.h"
#include "transient.h"
#include "require.h"

namespace celerity::algorithm
{

namespace detail
{

template <typename T, typename U>
auto link_transiently(T lhs, U rhs)
{
    using value_type = typename traits::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = traits::packaged_task_traits<T>::rank;

    transient_buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T, typename U>
auto link(T lhs, U rhs)
{
    using value_type = typename traits::packaged_task_traits<T>::output_value_type;
    constexpr auto rank = traits::packaged_task_traits<T>::rank;

    buffer<value_type, rank> out_buf{lhs.get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return sequence(t_left, t_right);
}

template <typename T,
          require<traits::is_sequence_v<T>,
                  traits::is_partially_packaged_task_v<traits::last_element_t<T>>,
                  traits::stage_requirement_v<traits::last_element_t<T>> == detail::stage_requirement::output> = yes>
auto terminate(T seq)
{
    using last_element_type = traits::last_element_t<T>;
    using traits = traits::packaged_task_traits<last_element_type>;

    using value_type = typename traits::output_value_type;
    constexpr auto rank = traits::rank;

    auto last = get_last_element(seq);

    buffer<value_type, rank> out_buf{last.get_range()};

    return append(remove_last_element(seq), last.complete(begin(out_buf), end(out_buf)));
}

} // namespace detail

template <typename T, typename U,
          require<traits::is_linkable_source_v<T>, traits::is_linkable_sink_v<U>> = yes>
auto operator|(T lhs, U rhs)
{
    using namespace traits;
    using namespace detail;

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

template <typename T, typename U,
          require<traits::is_sequence_v<T>,
                  traits::is_linkable_source_v<traits::last_element_t<T>>,
                  traits::is_linkable_sink_v<U>> = yes>
auto operator|(T lhs, U rhs)
{
    return detail::apply_append(lhs, rhs);
}

template <typename T, typename U,
          require<traits::is_packaged_task_v<T>,
                  traits::is_partially_packaged_task_v<U>,
                  traits::stage_requirement_v<U> == detail::stage_requirement::input> = yes>
auto operator|(T lhs, U rhs)
{
    auto t_right = rhs.complete(lhs.get_out_iterator(), lhs.get_out_iterator());

    return detail::sequence(lhs, t_right);
}

template <typename T, typename U,
          require<traits::is_sequence_v<T>,
                  traits::is_packaged_task_v<traits::last_element_t<T>>,
                  traits::is_partially_packaged_task_v<U>,
                  traits::stage_requirement_v<U> == detail::stage_requirement::input> = yes>
auto operator|(T lhs, U rhs)
{
    return detail::apply_append(lhs, rhs);
}

template <typename T, typename U,
          require<traits::is_linkable_source_v<traits::last_element_t<U>>,
                  traits::is_sequence_v<U>,
                  traits::is_linkable_sink_v<T>> = yes>
auto operator<<(T lhs, U rhs)
{
    using namespace detail;

    auto linked_sequence = link(get_last_element(rhs), lhs);

    auto linked_lhs = get_last_element(linked_sequence);
    auto linked_last_rhs = get_first_element(linked_sequence);

    return partial_t_joint{linked_lhs, append(remove_last_element(rhs), linked_last_rhs)};
}

template <typename T, typename U,
          require<traits::is_sequence_v<U>,
                  traits::is_linkable_source_v<traits::last_element_t<U>>,
                  traits::is_sequence_v<T>,
                  traits::is_linkable_sink_v<traits::last_element_t<T>>> = 0>
auto operator<<(T lhs, U rhs)
{
    using namespace detail;

    auto linked_sequence = link(get_last_element(rhs), get_last_element(lhs));

    auto linked_last_lhs = get_last_element(linked_sequence);
    auto linked_last_rhs = get_first_element(linked_sequence);

    partial_t_joint joint{linked_last_lhs, append(remove_last_element(rhs), linked_last_rhs)};

    return append(remove_last_element(lhs), joint);
}

template <typename T, typename U,
          require<algorithm::traits::is_linkable_sink_v<T>,
                  algorithm::traits::is_linkable_source_v<U>> = yes>
auto operator<<(T lhs, U rhs)
{
    return lhs << detail::sequence(rhs);
}

} // namespace celerity::algorithm

namespace celerity
{
template <typename T, int Rank, typename U,
          algorithm::require<algorithm::traits::is_linkable_sink_v<U>> = algorithm::yes>
auto operator|(celerity::buffer<T, Rank> lhs, U rhs)
{
    return rhs.complete(begin(lhs), end(lhs));
}

template <typename T, int Rank, typename U,
          algorithm::require<algorithm::traits::is_linkable_sink_v<T>> = algorithm::yes>
auto operator<<(T lhs, celerity::buffer<U, Rank> rhs)
{
    return lhs.complete(begin(rhs), end(rhs));
}

template <typename T, int Rank, typename U,
          algorithm::require<algorithm::traits::is_linkable_source_v<T>,
                             algorithm::traits::partially_packaged_task_traits<T>::requirement == algorithm::detail::stage_requirement::output> = algorithm::yes>
auto operator|(T lhs, celerity::buffer<U, Rank> rhs)
{
    return lhs.complete(begin(rhs), end(rhs));
}

template <typename T, int Rank, typename U,
          algorithm::require<algorithm::traits::is_sequence_v<T>,
                             algorithm::traits::is_linkable_source_v<algorithm::traits::last_element_t<T>>,
                             algorithm::traits::partially_packaged_task_traits<algorithm::traits::last_element_t<T>>::requirement == algorithm::detail::stage_requirement::output> = algorithm::yes>
auto operator|(T lhs, celerity::buffer<U, Rank> rhs)
{
    return algorithm::detail::apply_append(lhs, rhs);
}

} // namespace celerity

#endif // LINKAGE_H