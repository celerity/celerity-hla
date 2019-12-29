#ifndef GENERATE_H
#define GENERATE_H

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

template <typename ExecutionPolicy, typename F, typename T, int Rank>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    assert(beg.get_buffer().get_id() == end.get_buffer().get_id());

    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        if constexpr (algorithm::detail::get_accessor_type<F, 0>() == access_type::item)
        {
            return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(item); };
        }
        else
        {
            return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(); };
        }
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename F, typename T, int Rank,
          typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::item>>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return task<ExecutionPolicy>(detail::generate(p, beg, end, f));
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return scoped_sequence{actions::generate(p, beg, end, f), submit_to(p.q)};
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