#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity.h"
#include "task.h"

namespace celerity::algorithm
{
	auto submit_to(celerity::queue q)
	{
		return q;
	}

	template<template <typename...> typename Sequence, typename...Actions>
	auto operator | (Sequence<Actions...>&& lhs, celerity::queue& queue)
	{
		lhs(queue);
		return lhs;
	}

	template<template <typename...> typename Sequence, typename...Actions>
	auto operator | (Sequence<Actions...>&& lhs, celerity::queue&& queue)
	{
		std::invoke(lhs, queue);
		return lhs;
	}

	template<typename T, typename...Actions>
	auto operator | (task_t<T>&& lhs, celerity::queue&& queue)
	{
		std::invoke(lhs, queue);
		return lhs;
	}

	template<typename...Ts, typename...Us>
	auto operator | (task_t<Ts...> lhs, task_t<Us...> rhs)
	{
		return sequence<task_t<Ts...>, task_t<Us...>>{lhs, rhs};
	}

	template<typename T, typename U,
		std::enable_if_t<is_argless_invokable_v<T>&& is_argless_invokable_v<U>, int> = 0>
		auto operator | (T lhs, U rhs)
	{
		return sequence<T, U>{lhs, rhs};
	}

	template<typename T, typename U,
		std::enable_if_t<is_kernel_v<T> && !is_sequence_v<T>&& is_kernel_v<U>, int> = 0>
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
		return kernel_sequence<T..., U>{ { lhs.sequence(), rhs } };
	}

	template<typename T, typename U,
		std::enable_if_t<!is_kernel_v<T> && !is_sequence_v<T>&& is_kernel_v<U>, int> = 0>
		auto operator | (T lhs, U rhs)
	{
		return sequence<T, task_t<U>>{ lhs, { rhs }};
	}

	template<typename...Ts, typename U, size_t...Ids>
	auto unpack_kernel_sequence(kernel_sequence<Ts...> lhs, task_t<U> rhs, std::index_sequence<Ids...>)
	{
		sequence<task_t<Ts>...> seq{ task(std::get<Ids>(lhs.sequence().actions()))... };
		return sequence<task_t<Ts>..., task_t<U>>{ std::move(seq), rhs };
	}

	template<typename...Ts, typename U,
		std::enable_if_t<is_kernel_v<U>, int> = 0>
		auto operator | (kernel_sequence<Ts...> lhs, task_t<U> rhs)
	{
		return unpack_kernel_sequence(lhs, rhs, std::index_sequence_for<Ts...>{});
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
		std::enable_if_t<is_sequence_v<Sequence<Actions...>> && !is_kernel_v<Action>, int> = 0>
		auto operator | (Sequence<Actions...> && seq, Action action)
	{
		return sequence<Actions..., Action>{ std::move(seq), action };
	}

	template<template <typename...> typename Sequence, typename...Actions, typename Action,
		std::enable_if_t<is_sequence_v<Sequence<Actions...>>&& is_kernel_v<Action>, int> = 0>
		auto operator | (Sequence<Actions...> && seq, Action action)
	{
		return sequence<Actions..., task_t<Action>>{ std::move(seq), task(action) };
	}
}

#endif