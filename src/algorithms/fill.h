#ifndef FILL_H
#define FILL_H

#include "../iterator.h"
#include "../task.h"
#include "../task_sequence.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../scoped_sequence.h"
#include "../decoration.h"
#include "../placeholder.h"

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    assert(beg.get_buffer().get_id() == end.get_buffer().get_id());

    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);
        return [=](cl::sycl::item<Rank> item) { out_acc[item] = value; };
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename T>
auto fill(ExecutionPolicy p, buffer_iterator_placeholder, buffer_iterator_placeholder, const T &value)
{
    return [=](auto beg, auto end) {
        return decorate_generate<T>(task<ExecutionPolicy>(detail::fill(p, beg, end, value)), beg, end);
    };
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return scoped_sequence{actions::fill(p, beg, end, value), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T>
auto fill(ExecutionPolicy p, buffer_iterator_placeholder, buffer_iterator_placeholder, const T &value)
{
    return actions::fill(p, {}, {}, value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
    return fill(distr<KernelName>(q), beg, end, value);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer<T, Rank> in, const T &value)
{
    return fill(p, begin(in), end(in), value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer<T, Rank> in, const T &value)
{
    return fill(distr<KernelName>(q), begin(in), end(in), value);
}

template <typename KernelName, typename T>
auto fill(celerity::distr_queue q, buffer_placeholder, const T &value)
{
    return fill(distr<KernelName>(q), {}, {}, value);
}

} // namespace celerity::algorithm

#endif // FILL_H