#ifndef KERNEL_SEQUENCE_H
#define KERNEL_SEQUENCE_H

#include "celerity.h"

struct cgh{};
struct cgh_action { };

auto using_cgh() { return cgh_action{}; }

template<typename...Actions>
class kernel_sequence
{
public:
	kernel_sequence(sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }
    
	void operator()(distr_queue q) const 
  {
      // TODO
      // queue.submit([=](celerity::handler& cgh) {
      //std::invoke(invocable/*, cgh*/);
      sequence_(cgh{});
      // }
  }

private:
  sequence<Actions...> sequence_;
};

template<template <typename...> typename Sequence, typename...Actions>
kernel_sequence<Actions...> operator | (Sequence<Actions...>&& seq, const cgh_action&)
{
  return { std::move(seq) };
}

#endif