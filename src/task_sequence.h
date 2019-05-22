#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity.h"
#include "task.h"

template<typename F>
constexpr inline bool is_argless_invokable_v = ::is_invocable_v<F, void>;


auto submit_to(distr_queue q)
{
	return q;
}

template<template <typename...> typename Sequence, typename...Actions>
auto operator | (Sequence<Actions...>&& lhs, distr_queue& queue)
{
	lhs(queue);
	return lhs;
}

template<template <typename...> typename Sequence, typename...Actions>
auto operator | (Sequence<Actions...>&& lhs, distr_queue&& queue)
{
	lhs(queue);
	return lhs;
}

template<typename...Ts, typename...Us>
auto operator | (task_t<Ts...> lhs, task_t<Us...> rhs)
{
	return sequence<task_t<Ts...>, task_t<Us...>>{lhs, rhs};
}

template<typename T, typename U, 
	std::enable_if_t<is_argless_invokable_v<T> && is_argless_invokable_v<U>, int> = 0>
auto operator | (T lhs, U rhs)
{
	return sequence<T, U>{lhs, rhs};
}

template<typename T, typename U,
	std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T> && is_kernel_v<U>, int> = 0>
auto operator | (T lhs, U rhs)
{
	return kernel_sequence<T, U>{ { lhs, rhs }};
}

template<typename T, typename U,
	std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T> && !is_kernel_v<U>, int> = 0>
	auto operator | (T lhs, U rhs)
{
	return sequence<task_t<T>, U>{ { lhs }, rhs };
}

template<typename...T, typename U,
	std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator | (kernel_sequence<T...> lhs, U rhs)
{
	return kernel_sequence<T..., U>{ lhs.sequence() | rhs };
}

template<typename T, typename U,
	std::enable_if_t<!is_kernel_v<T> && !is_sequence_v<T> && is_kernel_v<U>, int> = 0>
	auto operator | (T lhs, U rhs)
{
	return sequence<T, task_t<U>>{ lhs, { rhs }};
}

template<typename...T, typename U, size_t...Ids>
auto unpack_kernel_sequence(kernel_sequence<T...> lhs, task_t<U> rhs, std::index_sequence<Ids...>)
{
	sequence<task_t<T>...> seq{ task(std::get<Ids>(lhs.sequence().actions()))... };
	return sequence<task_t<T>..., task_t<U>>{ std::move(seq) | rhs };
}

template<typename...T, typename U,
	std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator | (kernel_sequence<T...> lhs, task_t<U> rhs)
{
	return unpack_kernel_sequence(lhs, rhs, std::index_sequence_for<T...>{});
}

template<typename T, typename U>
auto operator | (task_t<T> lhs, U rhs)
{
	return sequence<task_t<T>, U>{lhs, rhs};
}

template<typename T, typename U,
	std::enable_if_t<is_kernel_v<U>, int> = 0>
auto operator | (task_t<T> lhs, U rhs)
{
	return sequence<task_t<T>, task_t<U>>{lhs, { rhs }};
}

template<typename T, typename U,
	std::enable_if_t<!is_kernel_v<U>, int> = 0>
	auto operator | (task_t<T> lhs, U rhs)
{
	return sequence<task_t<T>, U>{lhs, { rhs }};
}

template<typename T, typename U,
	std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T>, int> = 0>
auto operator | (T lhs, task_t<U> rhs)
{
	return sequence<task_t<T>, task_t<U>>{ { lhs }, rhs};
}

template<typename T, typename U,
	std::enable_if_t<!is_kernel_v<T> && !is_sequence_v<T>, int> = 0>
auto operator | (T lhs, task_t<U> rhs)
{
	return sequence<T, task_t<U>>{ { lhs }, rhs};
}

template<template <typename...> typename Sequence, typename...Actions, typename Action,
	typename = std::enable_if_t<is_sequence_v<Sequence<Actions...>>>>
auto operator | (Sequence<Actions...>&& seq, Action action)
{
	return sequence<Actions..., Action>{ std::move(seq), action };
}




#endif