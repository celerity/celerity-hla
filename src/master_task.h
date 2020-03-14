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

template <typename F, int... Ranks, typename... Ts, size_t... Is>
auto master_task(const F &f, std::tuple<buffer<Ts, Ranks>...> buffers, std::index_sequence<Is...>)
{
    using namespace traits;
    using namespace cl::sycl::access;

    using policy_type = detail::non_blocking_master_execution_policy;

    return master_task([=](auto &cgh) {
        const auto accessors = std::make_tuple(
            get_access<policy_type, mode::read, accessor_type_t<F, Is, Ts>>(cgh,
                                                                            begin(std::get<Is>(buffers)), end(std::get<Is>(buffers)))...);

        return [=]() {
            f((std::get<Is>(accessors)[cl::sycl::detail::make_item<Ranks>({}, {})])...);
        };
    });
}

} // namespace detail

template <typename ExecutionPolicy, typename F>
auto master_task(ExecutionPolicy p, const F &f)
{
    static_assert(std::is_same_v<detail::non_blocking_master_execution_policy, std::decay_t<ExecutionPolicy>>, "non-blocking master only");
    return std::invoke(detail::master_task(f), p.q);
}

template <typename ExecutionPolicy, typename F, typename... Buffers>
auto master_task(ExecutionPolicy p, const F &f, Buffers... buffers)
{
    const auto buffer_tuple = std::make_tuple(buffers...);
    return std::invoke(detail::master_task(f, buffer_tuple, std::index_sequence_for<Buffers...>{}), p.q);
}

} // namespace celerity::algorithm

#endif // MASTER_TASK_H
