#ifndef MASTER_TASK_H
#define MASTER_TASK_H

#include "policy.h"
#include "kernel_traits.h"
#include "task.h"

namespace celerity::algorithm
{

namespace detail
{

template <typename F>
auto master_task(const F &f)
{
    static_assert(algorithm::traits::is_master_task_v<F>, "not a master task");
    return task<non_blocking_master_execution_policy>(f);
}

} // namespace detail

template <typename ExecutionPolicy, typename F>
auto master_task(ExecutionPolicy p, const F &f)
{
    static_assert(std::is_same_v<detail::non_blocking_master_execution_policy, std::decay_t<ExecutionPolicy>>, "non-blocking master only");
    return std::invoke(detail::master_task(f), p.q);
}

} // namespace celerity::algorithm

#endif // MASTER_TASK_H
