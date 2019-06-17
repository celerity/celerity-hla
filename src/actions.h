#ifndef ACTIONS_H
#define ACTIONS_H

#include "sequence.h"
#include "celerity.h"

#include <mpi.h>

namespace celerity::algorithm::actions
{
	inline void global_barrier()
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}

	template<typename F, typename...Args>
	auto master_only(const F& f, Args&& ...args)
	{
		static thread_local const bool is_master = []()
		{
			if (int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS)
			{
				return world_rank == 0;
			}

			throw std::logic_error("MPI not initialized");
		}();

		if (!is_master) return;

		std::invoke(f, std::forward<Args>(args)...);
	}

	auto hello_world() { return []() { std::cout << "hello world" << std::endl; }; }

	auto incr(int& i) { return [&i]() { ++i; }; }

	auto with_queue() { return [](celerity::queue queue) { std::cout << "with queue" << std::endl; }; }

	struct dispatcher { };
	dispatcher dispatch() { return {}; }

	template<template <typename...> typename Sequence, typename...Actions>
	decltype(auto) operator | (Sequence<Actions...>&& sequence, const dispatcher&)
	{
		return std::invoke(sequence);
	}
}

#endif



