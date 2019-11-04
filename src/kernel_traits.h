#ifndef KERNEL_TRAITS_H
#define KERNEL_TRAITS_H

#include <type_traits>

#include "accessor_proxy.h"
#include "sequence_traits.h"

namespace celerity::algorithm
{
template <typename A, typename B>
struct is_combinable
	: std::integral_constant<bool, std::is_same<typename A::view_type, typename B::view_type>::value>
{
};

template <typename A, typename B>
constexpr inline bool is_combinable_v = is_combinable<A, B>::value;

namespace detail
{
template <int Rank>
using nd_kernel_arg_t = cl::sycl::item<Rank>;

using task_arg_t = handler &;

template <typename F>
using task_invoke_result_t = std::invoke_result_t<F, task_arg_t>;

template <typename F>
using task_first_invoke_result_t = std::tuple_element_t<0, std::invoke_result_t<F, task_arg_t>>;

template <typename F>
using task_second_invoke_result_t = std::tuple_element_t<1, std::invoke_result_t<F, task_arg_t>>;

template <typename F, int Rank>
struct is_nd_kernel : std::bool_constant<std::is_invocable_v<F, nd_kernel_arg_t<Rank>>>
{
};

template <typename F, int Rank>
constexpr inline bool is_nd_kernel_v = is_nd_kernel<F, Rank>::value;

template <typename F>
struct _is_kernel : std::bool_constant<
						is_nd_kernel_v<F, 1> ||
						is_nd_kernel_v<F, 2> ||
						is_nd_kernel_v<F, 3>>
{
};

template <typename F>
constexpr inline bool _is_kernel_v = _is_kernel<F>::value;

template <typename F>
struct _is_compute_task : std::bool_constant<
							  std::is_invocable_v<F, task_arg_t> &&
							  _is_kernel_v<task_invoke_result_t<F>>>
{
};

template <typename F>
constexpr inline bool _is_compute_task_v = _is_compute_task<F>::value;

template <typename F>
struct _is_master_task : std::bool_constant<
							 std::is_invocable_v<F, task_arg_t> &&
							 (_is_kernel_v<task_invoke_result_t<F>> || function_traits<task_invoke_result_t<F>>::arity == 0)>
{
};

template <typename F>
constexpr inline bool _is_master_task_v = _is_master_task<F>::value;

template <typename T>
struct _is_task_decorator : std::bool_constant<!is_sequence_v<T> && std::is_invocable_v<T, celerity::distr_queue &>>
{
};

template <typename F>
constexpr inline bool _is_task_decorator_v = _is_task_decorator<F>::value;

template <typename F, typename IteratorType>
struct _is_placeholder_task_impl : std::bool_constant<std::is_invocable_v<F, IteratorType, IteratorType>>
{
};

template <typename F, typename IteratorType>
struct _is_placeholder_task : std::conditional_t<is_sequence_v<F>,
												 std::false_type,
												 _is_placeholder_task_impl<F, IteratorType>>
{
};

template <typename F, typename IteratorType>
constexpr inline bool _is_placeholder_task_v = _is_placeholder_task<F, IteratorType>::value;

template <typename T, size_t... Is>
constexpr bool is_task_decorator_sequence_dispatch(std::index_sequence<Is...>)
{
	return ((_is_task_decorator_v<std::tuple_element_t<Is, typename T::actions_t>>)&&...);
}

template <typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
constexpr bool is_task_decorator_sequence()
{
	return is_task_decorator_sequence_dispatch<T>(std::make_index_sequence<T::num_actions>{});
}

template <typename T, std::enable_if_t<!is_sequence_v<T>, int> = 0>
constexpr bool is_task_decorator_sequence()
{
	return false;
}

} // namespace detail

} // namespace celerity::algorithm

#endif
