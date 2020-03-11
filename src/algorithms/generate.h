#ifndef GENERATE_H
#define GENERATE_H

#include "../iterator.h"
#include "../task.h"
#include "../task_sequence.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"
namespace celerity::algorithm
{
namespace actions
{
namespace detail
{

template <typename ExecutionPolicy, typename F, template <typename, int> typename IteratorType, typename T, int Rank>
auto generate(IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        if constexpr (algorithm::detail::arity_v<F> == 1)
        {
            return [=](item_context<Rank, T> &ctx) {
                out_acc[ctx[0]] = f(ctx[0]);
            };
        }
        else
        {
            return [=](item_context<Rank, T> &ctx) {
                out_acc[ctx[0]] = f();
            };
        }
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename F, typename T, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 1, int> = 0>
auto generate(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(algorithm::detail::get_accessor_type<F, 0>() == access_type::item);

    const auto t = task<ExecutionPolicy>(detail::generate<ExecutionPolicy>(beg, end, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 1, int> = 0>
auto generate(cl::sycl::range<Rank> range, const F &f)
{
    static_assert(algorithm::detail::get_accessor_type<F, 0>() == access_type::item);

    using value_type = std::invoke_result_t<F, cl::sycl::item<Rank>>;

    return package_generate<value_type>(
        [f](auto beg, auto end) { return task<ExecutionPolicy>(detail::generate<ExecutionPolicy>(beg, end, f)); },
        range);
}

// DISABLED
template <typename ExecutionPolicy, typename F, typename T, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 0, int> = 0>
auto generate(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(std::is_void_v<F>,
                  "Disabled as there is no real use cases as long as functors are required to be immutable");

    using value_type = std::invoke_result_t<F>;
    return package_generate<value_type>(task<ExecutionPolicy>(detail::generate<ExecutionPolicy>(beg, end, f)), beg, end);
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return std::invoke(actions::generate<ExecutionPolicy>(beg, end, f), p.q);
}

template <typename KernelName, typename F, int Rank>
auto generate(cl::sycl::range<Rank> range, const F &f)
{
    using execution_policy = named_distributed_execution_policy<KernelName>;
    return actions::generate<execution_policy>(range, f);
}

template <typename KernelName, typename T, int Rank, typename F>
auto generate(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return generate(distr<KernelName>(q), beg, end, f);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
    return generate(p, begin(in), end(in), f);
}

template <typename KernelName, typename T, int Rank, typename F>
auto generate(celerity::distr_queue q, buffer<T, Rank> in, const F &f)
{
    return generate(distr<KernelName>(q), begin(in), end(in), f);
}

} // namespace celerity::algorithm

#endif // GENERATE_H