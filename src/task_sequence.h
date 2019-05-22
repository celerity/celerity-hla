#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity.h"
#include "task.h"

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


template<typename T, typename U, typename = std::enable_if_t<!std::is_convertible<T, task_t<T>>::value && !std::is_convertible<U, task_t<U>>::value>>
auto operator | (T lhs, U rhs)
{
	return sequence<T, U>{lhs, rhs};
}

template<typename T, typename U>
auto operator | (task_t<T> lhs, U rhs)
{
	return sequence<task_t<T>, U>{lhs, rhs};
}


template<template <typename...> typename Sequence, typename...Actions, typename Action,
	typename = std::enable_if_t<is_sequence_v<Sequence<Actions...>>>>
	auto operator | (Sequence<Actions...>&& seq, Action action)
{
	return sequence<Actions..., Action>{ std::move(seq), action };
}


template<typename T, typename U,
	typename = std::enable_if_t<!is_sequence_v<T>>>
	auto operator | (T lhs, task_t<U> rhs)
{
	return sequence<T, task_t<U>>{lhs, rhs};
}

#endif