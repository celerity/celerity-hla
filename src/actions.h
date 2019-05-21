
#ifndef ACTIONS_H
#define ACTIONS_H

#include "sequence.h"
#include "kernel_sequence.h"
#include "celerity.h"

auto hello_world() { return []() { std::cout << "hello world" << std::endl; }; }

auto incr(int& i) { return [&i]() { ++i; }; }

auto with_queue() { return [](distr_queue queue) { std::cout << "with queue" << std::endl; }; }

struct dispatcher { };
dispatcher dispatch() { return {}; }

template<template <typename...> typename Sequence, typename...Actions>
decltype(auto) operator | (Sequence<Actions...>&& sequence, const dispatcher& dispatcher)
{
	//TODO return std::invoke(sequence);
  return sequence();
}

#endif



