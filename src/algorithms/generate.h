#ifndef GENERATE_H
#define GENERATE_H

#include "../iterator.h"
#include "../task.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"
#include "../require.h"

namespace celerity::algorithm
{

namespace detail
{

template <typename ExecutionPolicy, typename F, template <typename, int> typename IteratorType, typename T, int Rank>
auto generate_impl(IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const F &f)
{
    using namespace traits;
    using namespace cl::sycl::access;

    using policy_type = strip_queue_t<ExecutionPolicy>;

    return [=](celerity::handler &cgh) {
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

        return [=](item_context<Rank, T()> &ctx) {
            if constexpr (traits::arity_v<F> == 1)
            {
                out_acc[ctx.get_out()] = f(ctx.get_item());
            }
            else
            {
                out_acc[ctx.get_out()] = f();
            }
        };
    };
}

template <typename ExecutionPolicy, typename F, typename T, int Rank,
          require<traits::arity_v<F> == 1> = yes>
auto generate(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(traits::get_accessor_type<F, 0>() == access_type::item);

    const auto t = task<ExecutionPolicy>(generate_impl<ExecutionPolicy>(beg, end, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F, int Rank,
          require<traits::arity_v<F> == 1> = yes>
auto generate(cl::sycl::range<Rank> range, const F &f)
{
    static_assert(traits::get_accessor_type<F, 0>() == access_type::item);

    using value_type = std::invoke_result_t<F, cl::sycl::item<Rank>>;

    return package_generate<value_type>(
        [f](auto beg, auto end) { return task<ExecutionPolicy>(generate_impl<ExecutionPolicy>(beg, end, f)); },
        range);
}

// DISABLED
template <typename ExecutionPolicy, typename F, typename T, int Rank,
          require<traits::arity_v<F> == 0> = yes>
auto generate(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    static_assert(std::is_void_v<F>,
                  "Disabled as there is no real use cases as long as functors are required to be immutable");

    using value_type = std::invoke_result_t<F>;
    return package_generate<value_type>(task<ExecutionPolicy>(generate_impl<ExecutionPolicy>(beg, end, f)), beg, end);
}

} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return std::invoke(detail::generate<ExecutionPolicy>(beg, end, f), p.q);
}

template <typename KernelName, typename F, int Rank>
auto generate_n(cl::sycl::range<Rank> range, const F &f)
{
    using execution_policy = detail::named_distributed_execution_policy<KernelName>;
    return detail::generate<execution_policy>(range, f);
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