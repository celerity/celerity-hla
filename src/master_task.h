#ifndef MASTER_TASK_H
#define MASTER_TASK_H

#include "policy.h"
#include "kernel_traits.h"

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{

} // namespace detail

template <typename F>
auto master_task(const F &f)
{
    static_assert(algorithm::detail::is_master_task_v<F>, "not a master task");
    return task<non_blocking_master_execution_policy>(f);
}

} // namespace actions

template <typename ExecutionPolicy, typename F>
auto master_task(ExecutionPolicy p, const F &f)
{
    static_assert(std::is_same_v<non_blocking_master_execution_policy, std::decay_t<ExecutionPolicy>>, "non-blocking master only");
    return scoped_sequence{actions::master_task(f), submit_to(p.q)};
}

} // namespace celerity::algorithm

#endif // MASTER_TASK_H
