#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity.h"

template<template <typename...> typename Sequence, typename...Actions>
class task_sequence
{
public:
	task_sequence(distr_queue q, Sequence<Actions...>&& s)
		: queue_(q), sequence_(std::move(s)) { }

	void operator()() const 
  {
      sequence_(queue_);
  }

private:
	distr_queue queue_;
  Sequence<Actions...> sequence_;
};

auto submit_to(distr_queue q)
{
	return q;
}

template<template <typename...> typename Sequence, typename...Actions>
task_sequence<Sequence, Actions...> operator | (Sequence<Actions...>&& seq, const distr_queue& queue)
{
	//return std::invoke(sequence);
  task_sequence<Sequence, Actions...> qs { queue, std::move(seq) };
  qs();
  return qs;
}

#endif