#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity_helper.h"
#include "task.h"
#include "packaged_task.h"

namespace celerity::algorithm
{

inline auto submit_to(celerity::distr_queue q)
{
	return q;
}

template <typename F>
decltype(auto) operator|(task_t<non_blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
{
	return std::invoke(lhs, queue);
}

template <typename F>
decltype(auto) operator|(task_t<blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T>, int> = 0>
decltype(auto) operator|(T &&lhs, distr_queue &&queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T>, int> = 0>
decltype(auto) operator|(T &&lhs, distr_queue &queue)
{
	return std::invoke(lhs, queue);
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T>, int> = 0>
decltype(auto) operator|(T &lhs, distr_queue &queue)
{
	return std::invoke(lhs, queue);
}

template <typename LhsExecutionPolicy, typename RhsExecutionPolicy, typename... Ts, typename... Us>
auto operator|(task_t<LhsExecutionPolicy, Ts...> lhs, task_t<RhsExecutionPolicy, Us...> rhs)
{
	return sequence<task_t<LhsExecutionPolicy, Ts...>, task_t<RhsExecutionPolicy, Us...>>{lhs, rhs};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<detail::is_compute_task_v<T> && !is_sequence_v<T>, int> = 0>
auto operator|(T lhs, task_t<ExecutionPolicy, U> rhs)
{
	return sequence<task_t<ExecutionPolicy, T>, task_t<ExecutionPolicy, U>>{{lhs}, rhs};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<!detail::is_compute_task_v<T> && !is_sequence_v<T> && !is_task_v<T>, int> = 0>
auto operator|(T lhs, task_t<ExecutionPolicy, U> rhs)
{
	return sequence<T, task_t<ExecutionPolicy, U>>{{lhs}, rhs};
}

} // namespace celerity::algorithm

#endif