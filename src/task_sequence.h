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

template <template <typename...> typename Sequence, typename... Actions>
decltype(auto) operator|(Sequence<Actions...> &&lhs, celerity::distr_queue &queue)
{
	return std::invoke(lhs, queue);
}

template <template <typename...> typename Sequence, typename... Actions>
decltype(auto) operator|(Sequence<Actions...> &&lhs, celerity::distr_queue &&queue)
{
	return std::invoke(lhs, queue);
}

template <typename ExecutionPolicy, typename T, typename... Actions>
decltype(auto) operator|(task_t<ExecutionPolicy, T> &&lhs, celerity::distr_queue &&queue)
{
	return std::invoke(lhs, queue);
}

template <typename ExecutionPolicy, typename... Ts, typename... Us>
auto operator|(task_t<ExecutionPolicy, Ts...> lhs, task_t<ExecutionPolicy, Us...> rhs)
{
	return sequence<task_t<ExecutionPolicy, Ts...>, task_t<ExecutionPolicy, Us...>>{lhs, rhs};
}

template <typename T, typename U,
		  std::enable_if_t<is_argless_invokable_v<T> && is_argless_invokable_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
	return sequence<T, U>{lhs, rhs};
}

template <typename T, typename U,
		  std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T> && is_kernel_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
	return kernel_sequence<T, U>{{lhs, rhs}};
}

template <typename T, typename U,
		  std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T> && !is_kernel_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
	return sequence<task_t<distributed_execution_policy, T>, U>{{lhs}, rhs};
}

template <typename... T, typename U,
		  std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator|(kernel_sequence<T...> lhs, U rhs)
{
	return kernel_sequence<T..., U>{{lhs.get_sequence(), rhs}};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<!is_kernel_v<T> && !is_sequence_v<T> && is_kernel_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
	return sequence<T, task_t<ExecutionPolicy, U>>{lhs, {rhs}};
}

template <typename ExecutionPolicy, typename... Ts, typename U, size_t... Ids>
auto unpack_kernel_sequence(kernel_sequence<Ts...> lhs, task_t<ExecutionPolicy, U> rhs, std::index_sequence<Ids...>)
{
	sequence<task_t<ExecutionPolicy, Ts>...> seq{task(std::get<Ids>(lhs.get_sequence().actions()))...};
	return sequence<task_t<ExecutionPolicy, Ts>..., task_t<ExecutionPolicy, U>>{std::move(seq), rhs};
}

template <typename ExecutionPolicy, typename... Ts, typename U,
		  std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator|(kernel_sequence<Ts...> lhs, task_t<ExecutionPolicy, U> rhs)
{
	return unpack_kernel_sequence(lhs, rhs, std::index_sequence_for<Ts...>{});
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<!is_task_v<U>, int> = 0>
auto operator|(task_t<ExecutionPolicy, T> lhs, U rhs)
{
	return sequence<task_t<ExecutionPolicy, T>, U>{lhs, rhs};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator|(task_t<ExecutionPolicy, T> lhs, U rhs)
{
	return sequence<task_t<ExecutionPolicy, T>, task_t<ExecutionPolicy, U>>{lhs, {rhs}};
}

template <typename ExecutionPolicy, typename T, typename U,
		  std::enable_if_t<!is_kernel_v<U> && !is_task_v<U>, int> = 0>
auto operator|(task_t<ExecutionPolicy, T> lhs, U rhs)
{
	return sequence<task_t<ExecutionPolicy, T>, U>{lhs, {rhs}};
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

template <template <typename...> typename Sequence, typename... Actions, typename Action,
		  std::enable_if_t<is_sequence_v<Sequence<Actions...>> && !is_kernel_v<Action>, int> = 0>
auto operator|(Sequence<Actions...> &&seq, Action action)
{
	return sequence<Actions..., Action>{std::move(seq), action};
}

template <template <typename...> typename Sequence, typename... Actions, typename Action,
		  std::enable_if_t<is_sequence_v<Sequence<Actions...>> && is_kernel_v<Action>, int> = 0>
auto operator|(Sequence<Actions...> &&seq, Action action)
{
	return sequence<Actions..., task_t<distributed_execution_policy, Action>>{std::move(seq), task(action)};
}
} // namespace celerity::algorithm

#endif