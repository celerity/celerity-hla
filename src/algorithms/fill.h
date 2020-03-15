#ifndef FILL_H
#define FILL_H

#include "../iterator.h"
#include "../task.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"

namespace celerity::algorithm
{
namespace detail
{

template <typename ExecutionPolicy, template <typename, int> typename IteratorType, typename T, int Rank>
auto fill_impl(IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const T &value)
{
    using namespace traits;
    using namespace cl::sycl::access;

    using policy_type = strip_queue_t<ExecutionPolicy>;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = value;
        };
    };
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    const auto t = task<ExecutionPolicy>(fill_impl<ExecutionPolicy>(beg, end, value));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(cl::sycl::range<Rank> range, const T &value)
{
    return package_generate<T>(
        [value](auto beg, auto end) { return task<ExecutionPolicy>(fill_impl<ExecutionPolicy>(beg, end, value)); },
        range);
}

} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return std::invoke(detail::fill<ExecutionPolicy>(beg, end, value), p.q);
}

template <typename KernelName, typename T, int Rank>
auto fill_n(cl::sycl::range<Rank> range, const T &value)
{
    using execution_policy = detail::named_distributed_execution_policy<KernelName>;
    return detail::fill<execution_policy>(range, value);
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