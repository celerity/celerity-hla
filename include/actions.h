#ifndef ACTIONS_H
#define ACTIONS_H

#include "sequence.h"
#include "celerity_helper.h"

#include <mpi.h>
#include <iostream>
#include <stdexcept>

namespace celerity::hla
{
	// unused
	inline void sync()
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// TODO: rename this so it does not seem like it submits a with_master_access task...
	template <typename F, typename... Args>
	auto on_master(F &&f, Args &&...args)
	{
		static thread_local const bool is_master = []() {
			if (int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS)
			{
				return world_rank == 0;
			}

			throw std::logic_error("MPI not initialized");
		}();

		if (!is_master)
			return;

		std::invoke(f, std::forward<Args>(args)...);
	}

} // namespace celerity::hla

#endif
