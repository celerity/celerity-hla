#ifndef DECORATED_TASK_H
#define DECORATED_TASK_H

#include "iterator.h"
#include "celerity_helper.h"

#include "computation_type.h"

#include "decorator_traits.h"

#include "decorators/transform_decorator.h"
#include "decorators/generate_decorator.h"
#include "decorators/zip_decorator.h"

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
template <typename T, int Rank, typename U, std::enable_if_t<algorithm::detail::is_placeholder_task_v<U, algorithm::buffer_iterator<T, Rank>>, int> = 0>
auto operator|(celerity::buffer<T, Rank> &lhs, U rhs)
{
    return rhs(begin(lhs), end(lhs));
}

template <typename T, int Rank, typename U, std::enable_if_t<algorithm::detail::is_placeholder_task_v<U, algorithm::buffer_iterator<T, Rank>>, int> = 0>
auto operator<<(U lhs, celerity::buffer<T, Rank> &rhs)
{
    return lhs(begin(rhs), end(rhs));
}
} // namespace celerity

#endif // DECORATED_TASK_H