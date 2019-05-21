#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity.h"
#include "task.h"

auto submit_to(distr_queue q)
{
	return q;
}

template<typename...Ts, typename...Us>
auto operator | (task_t<Ts...>&& lhs, task_t<Us...>&& rhs)
{
  return sequence<task_t<Ts...>, task_t<Us...>>{lhs, rhs};
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

#endif