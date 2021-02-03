#ifndef MASTER_TASK_H
#define MASTER_TASK_H

#include "policy.h"
#include "kernel_traits.h"
#include "task.h"

namespace celerity::hla
{

    namespace detail
    {

        template <typename F>
        auto master_task(const F &f)
        {
            static_assert(hla::traits::is_master_task_v<F>, "not a master task");
            return task<non_blocking_master_execution_policy>(f);
        }

        template <typename F, int... Ranks, typename... Ts, size_t... Is>
        auto master_task(const F &f, std::tuple<buffer<Ts, Ranks>...> buffers, std::index_sequence<Is...>)
        {
            using namespace traits;
            using namespace cl::sycl::access;

            using policy_type = detail::non_blocking_master_execution_policy;

            static_assert(((traits::is_all_v<traits::arg_type_t<F, Is>>)&&...), "only all<> accessors supported");

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

    template <typename ExecutionPolicy, typename F, int... Ranks, typename... Ts>
    auto master_task(ExecutionPolicy p, std::tuple<buffer<Ts, Ranks>...> buffers, const F &f)
    {
        constexpr auto buffer_count = std::tuple_size_v<std::tuple<buffer<Ts, Ranks>...>>;

        static_assert(buffer_count == traits::arity_v<F>, "kernel needs to take the same number of arguments as there are buffers");

        return std::invoke(detail::master_task(f, buffers, std::make_index_sequence<buffer_count>{}),
                           p.q);
    }

    template <typename ExecutionPolicy, typename F, int Rank, typename T>
    auto master_task(ExecutionPolicy p, buffer<T, Rank> buf, const F &f)
    {
        return master_task(p, std::make_tuple(buf), f);
    }

    template <int... Ranks, typename... Ts>
    auto pack(buffer<Ts, Ranks>... buffers)
    {
        return std::tuple{buffers...};
    }

} // namespace celerity::hla

#endif // MASTER_TASK_H
