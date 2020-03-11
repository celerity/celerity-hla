#ifndef FILL_H
#define FILL_H

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

template <typename ExecutionPolicy, template <typename, int> typename IteratorType, typename T, int Rank>
auto fill(IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const T &value)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = value;
        };
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    const auto t = task<ExecutionPolicy>(detail::fill<ExecutionPolicy>(beg, end, value));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(cl::sycl::range<Rank> range, const T &value)
{
    return package_generate<T>(
        [value](auto beg, auto end) { return task<ExecutionPolicy>(detail::fill<ExecutionPolicy>(beg, end, value)); },
        range);
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return std::invoke(actions::fill<ExecutionPolicy>(beg, end, value), p.q);
}

template <typename KernelName, typename T, int Rank>
auto fill(cl::sycl::range<Rank> range, const T &value)
{
    using execution_policy = named_distributed_execution_policy<KernelName>;
    return actions::fill<execution_policy>(range, value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return fill(distr<KernelName>(q), beg, end, value);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer<T, Rank> out, const T &value)
{
    return fill(p, begin(out), end(out), value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer<T, Rank> in, const T &value)
{
    return fill(distr<KernelName>(q), begin(in), end(in), value);
}

} // namespace celerity::algorithm

#endif // FILL_H