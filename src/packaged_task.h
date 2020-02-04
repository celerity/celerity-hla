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

namespace celerity::algorithm
{

template <typename Placeholder, typename Iterator>
inline constexpr auto is_compatible_placeholder_v = detail::is_placeholder_task_v<Placeholder, Iterator>;

template <typename Placeholder, typename Iterator>
using substitution_result_t = std::invoke_result_t<Placeholder, Iterator>;

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && is_compatible_placeholder_v<U, typename T::output_iterator_type> && detail::is_packaged_task_v<substitution_result_t<U, typename T::output_iterator_type>>, int> = 0>
auto operator|(T lhs, U rhs)
{
    const auto output_it = lhs.get_out_iterator();
    const auto r = rhs(begin(output_it.get_buffer()), end(output_it.get_buffer()));
    return lhs | r;
}

// TODO
//
// placholder substitution results in another placholder -> for transform tasks this is okay
// output buffer will be created on the fly
//
// zip tasks need distinction between missing second operand and missing output buffer
//
// creating output buffers requires additional type information from the result of the
// kernel functor -> create partially packaged tasks class
template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && is_compatible_placeholder_v<U, typename T::output_iterator_type> && !detail::is_packaged_task_v<substitution_result_t<U, typename T::output_iterator_type>>, int> = 0>
auto operator|(T lhs, U rhs)
{

    const auto output_it = lhs.get_out_iterator();
    const auto r = rhs(begin(output_it.get_buffer()), end(output_it.get_buffer()));
    return lhs | r;
}

template <typename T, typename U, std::enable_if_t<detail::is_partially_packaged_task_v<T> && detail::stage_requirement_v<T> == stage_requirement::output && detail::is_partially_packaged_task_v<U> && detail::stage_requirement_v<U> == stage_requirement::input, int> = 0>
auto operator|(T lhs, U rhs)
{
    using value_type = typename T::output_value_type;

    // TODO: should honor the actual computation range
    buffer<value_type, T::rank> out_buf{lhs.get_in_beg().get_buffer().get_range()};

    auto t_left = lhs.complete(begin(out_buf), end(out_buf));
    auto t_right = rhs.complete(begin(out_buf), end(out_buf));

    return t_left | t_right;
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && detail::is_partially_packaged_task_v<U> && detail::stage_requirement_v<U> == stage_requirement::output, int> = 0>
auto operator|(T lhs, U rhs)
{
    using value_type = typename T::output_value_type;

    // TODO: should honor the actual computation range
    buffer<value_type, T::rank> out_buf{rhs.get_in_beg().get_buffer().get_range()};
    auto t = rhs.complete(begin(out_buf), end(out_buf));

    return lhs | t;
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_v<T> && detail::is_packaged_task_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return sequence(lhs, rhs);
}

// template <typename T, typename U,
//     std::enable_if_t<detail::is_packaged_task_v<T> &&
//     detail::is_packaged_task_v<U> && !detail::computation_type_of_v<U, computation_type::generate>, int> = 0>
// auto operator|(T lhs, U rhs)
// {
//     return sequence(lhs, rhs);
// }

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_sequence<T>() && detail::is_packaged_task_v<U> && !detail::computation_type_of_v<U, computation_type::generate>, int> = 0>
auto operator|(T lhs, U rhs)
{
    return lhs | sequence(rhs);
}

template <typename T, typename U, std::enable_if_t<detail::is_packaged_task_sequence<T>() && detail::is_placeholder_task_v<U, typename last_element_t<T>::output_iterator_type>, int> = 0>
auto operator|(T lhs, U rhs)
{
    auto last = get_last_element(lhs);

    const auto output_it = last.get_out_iterator();
    const auto r = rhs(begin(output_it.get_buffer()), end(output_it.get_buffer()));

    return lhs | r;
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

template <typename T, int Rank, typename U, std::enable_if_t<algorithm::detail::is_placeholder_task_v<U, algorithm::buffer_iterator<T, Rank>>, int> = 0>
auto operator<<(U lhs, celerity::buffer<T, Rank> &rhs)
{
    return lhs(begin(rhs), end(rhs));
}
} // namespace celerity

#endif // PACKAGED_TASK_H