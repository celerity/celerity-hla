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
      // TODO: std::invoke(queue_);
      sequence_(queue_);
  }

private:
	distr_queue queue_;
  Sequence<Actions...> sequence_;
};

template<template <typename...> typename Sequence, typename...Actions>
task_sequence<Sequence, Actions...> operator | (Sequence<Actions...>&& seq, const distr_queue& queue)
{
  task_sequence<Sequence, Actions...> qs { queue, std::move(seq) };
  // TODO: std::invoke(qs);
  qs();
  return qs;
}

#endif