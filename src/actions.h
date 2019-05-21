
#ifndef ACTIONS_H
#define ACTIONS_H

#include "sequence.h"
#include "kernel_sequence.h"
#include "celerity.h"

auto hello_world() { return []() { std::cout << "hello world" << std::endl; }; }

auto incr(int& i) { return [&i]() { ++i; }; }

template<typename T>
auto task(const T& invocable)
{ 
  using sequence_type = sequence<decltype(invocable)>;
  return kernel_sequence<sequence_type>{sequence_type{invocable}};
}

auto with_queue() { return [](distr_queue queue) { std::cout << "with queue" << std::endl; }; }

auto submit_to(distr_queue q)
{
	return q;
}

struct dispatcher { };
dispatcher dispatch() { return {}; }

template<template <typename...> typename Sequence, typename...Actions>
decltype(auto) operator | (Sequence<Actions...>&& sequence, const dispatcher& dispatcher)
{
	//TODO return std::invoke(sequence);
  return sequence();
}

#endif



