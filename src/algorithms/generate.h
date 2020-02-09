#ifndef GENERATE_H
#define GENERATE_H

#include "../iterator.h"
#include "../task.h"
#include "../task_sequence.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../scoped_sequence.h"
#include "../packaged_task.h"
#include "../placeholder.h"
#include "../fusion.h"

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{

template <typename ExecutionPolicy, typename F, template <typename, int> typename IteratorType, typename T, int Rank>
auto generate(ExecutionPolicy p, IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        if constexpr (algorithm::detail::arity_v<F> == 1)
        {
            return [=](item_context<Rank, T>& ctx) { out_acc[ctx] = f(ctx); };
        }
        else
        {
            return [=](item_context<Rank, T>& ctx) { out_acc[ctx] = f(); };
        }
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename F, typename T, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 1, int> = 0>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(algorithm::detail::get_accessor_type<F, 0>() == access_type::item);

    using value_type = std::invoke_result_t<F, cl::sycl::item<Rank>>;

    return package_generate<value_type>(
        [p, f](auto _beg, auto _end) { return task<ExecutionPolicy>(detail::generate(p, _beg, _end, f)); },
        beg, end);
}

template <typename ExecutionPolicy, typename F, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 1, int> = 0>
auto generate(ExecutionPolicy p, cl::sycl::range<Rank> range,  const F &f)
{
    static_assert(algorithm::detail::get_accessor_type<F, 0>() == access_type::item);

    using value_type = std::invoke_result_t<F, cl::sycl::item<Rank>>;
    
    return package_generate<value_type>(
        [p, f](auto beg, auto end) { return task<ExecutionPolicy>(detail::generate(p, beg, end, f)); },
        range);
}

// DISABLED
template <typename ExecutionPolicy, typename F, typename T, int Rank,
          std::enable_if_t<algorithm::detail::arity_v<F> == 0, int> = 0>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(std::is_void_v<F>,
                  "Disabled as there is no real use cases as long as functors are required to be immutable");

    using value_type = std::invoke_result_t<F>;
    return package_generate<value_type>(task<ExecutionPolicy>(detail::generate(p, beg, end, f)), beg, end);
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

template <typename KernelName, typename F, int Rank>
auto generate(celerity::distr_queue q, cl::sycl::range<Rank> range, const F &f)
{
    return actions::generate(distr<KernelName>(q), range, f);
}

} // namespace celerity::algorithm

#endif // GENERATE_H