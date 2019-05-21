#ifndef KERNEL_SEQUENCE_H
#define KERNEL_SEQUENCE_H

#include "celerity.h"

template<typename...Actions>
class kernel_sequence
{
public:
	kernel_sequence(sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }
    
	void operator()(distr_queue q) const 
  {
      q.submit([&](auto cgh) { sequence_(cgh); });
  }

  auto& sequence() { return sequence_; }

private:
  ::sequence<Actions...> sequence_;
};

template<template <typename...> typename Sequence, typename...Actions>
kernel_sequence<Actions...> task(Sequence<Actions...>&& seq)
{
  return { std::move(seq) };
}

template<typename T>
auto task(const T& invocable)
{ 
  using sequence_type = sequence<decltype(invocable)>;
  return kernel_sequence<sequence_type>{sequence_type{invocable}};
}

template<typename...Ts, typename...Us>
auto operator | (kernel_sequence<Ts...>&& lhs, kernel_sequence<Us...>&& rhs)
{
  auto seq = lhs.sequence() | rhs.sequence();
	return kernel_sequence<decltype(seq)>{ seq };
}

#endif