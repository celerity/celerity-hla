#ifndef TASK_SEQUENCE_H
#define TASK_SEQUENCE_H

#include "celerity_helper.h"
#include "task.h"

namespace celerity::algorithm
{

inline auto submit_to(celerity::distr_queue q)
{
	return q;
}

template <typename F>
decltype(auto) operator|(task_t<non_blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
{
	return std::invoke(lhs, queue);
}

template <typename F>
decltype(auto) operator|(task_t<blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
{
	return std::invoke(lhs, queue);
}

} // namespace celerity::algorithm

#endif