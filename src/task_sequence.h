#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity_helper.h"
#include "task.h"

namespace celerity::algorithm
{

inline auto submit_to(celerity::distr_queue q)
{
	return q;
}

template <typename T, std::enable_if_t<detail::_is_task_decorator_v<T>, int> = 0>
decltype(auto) operator|(T &&lhs, celerity::distr_queue &&queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::_is_task_decorator_v<T>, int> = 0>
decltype(auto) operator|(T &&lhs, celerity::distr_queue &queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::_is_task_decorator_v<T>, int> = 0>
decltype(auto) operator|(T &lhs, celerity::distr_queue &&queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::_is_task_decorator_v<T>, int> = 0>
decltype(auto) operator|(T &lhs, celerity::distr_queue &queue)
{
	return std::invoke(lhs, queue);
}

template <typename LhsExecutionPolicy, typename RhsExecutionPolicy, typename... Ts, typename... Us>
auto operator|(task_t<LhsExecutionPolicy, Ts...> lhs, task_t<RhsExecutionPolicy, Us...> rhs)
{
	return sequence<task_t<LhsExecutionPolicy, Ts...>, task_t<RhsExecutionPolicy, Us...>>{lhs, rhs};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T>, int> = 0>
auto operator|(T lhs, task_t<ExecutionPolicy, U> rhs)
{
	return sequence<task_t<ExecutionPolicy, T>, task_t<ExecutionPolicy, U>>{{lhs}, rhs};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<!is_kernel_v<T> && !is_sequence_v<T> && !is_task_v<T>, int> = 0>
auto operator|(T lhs, task_t<ExecutionPolicy, U> rhs)
{
	return sequence<T, task_t<ExecutionPolicy, U>>{{lhs}, rhs};
}

} // namespace celerity::algorithm

#endif