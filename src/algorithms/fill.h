#ifndef FILL_H
#define FILL_H

#include "../iterator.h"
#include "../task.h"
#include "../task_sequence.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../scoped_sequence.h"
#include "../packaged_task.h"
#include "../placeholder.h"

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{

template <typename ExecutionPolicy, template <typename, int> typename IteratorType, typename T, int Rank>
auto fill(ExecutionPolicy p, IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const T &value)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);
        return [=](item_context<Rank, T> &ctx) { out_acc[ctx] = value; };
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return package_generate<T>(
        [p, value](auto _beg, auto _end) { return task<ExecutionPolicy>(detail::fill(p, _beg, _end, value)); },
        beg, end);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, cl::sycl::range<Rank> range, const T &value)
{
    return package_generate<T>(
        [p, value](auto beg, auto end) { return task<ExecutionPolicy>(detail::fill(p, beg, end, value)); },
        range);
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return scoped_sequence{actions::fill(p, beg, end, value), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, cl::sycl::range<Rank> range, const T &value)
{
    return actions::fill(p, range, value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, cl::sycl::range<Rank> range, const T &value)
{
    return fill(distr<KernelName>(q), range, value);
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