#ifndef CELERITY_HLA_GENERATE_H
#define CELERITY_HLA_GENERATE_H

#include "../../iterator.h"
#include "../../task.h"
#include "../../policy.h"
#include "../../sequencing.h"
#include "../../require.h"

#include "../accessor_proxies.h"
#include "../packaged_tasks/packaged_generate.h"

using celerity::algorithm::buffer_iterator;

namespace celerity::hla::experimental
{
    namespace detail
    {
        template <typename ExecutionPolicy, typename F, template <typename, int> typename IteratorType, typename T, int Rank>
        auto generate_impl(IteratorType<T, Rank> beg, IteratorType<T, Rank> end, const F &f)
        {
            using namespace cl::sycl::access;

            using policy_type = algorithm::traits::strip_queue_t<ExecutionPolicy>;

            return [=](celerity::handler &cgh) {
                auto out_acc = get_out_access<policy_type, mode::discard_write>(cgh, beg, end);

                return [=](algorithm::detail::item_context<Rank, T()> &ctx) {
                    if constexpr (algorithm::traits::arity_v<F> == 1)
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
                  celerity::algorithm::require<celerity::algorithm::traits::arity_v<F> == 1> = celerity::algorithm::yes>
        auto generate(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
        {
            //static_assert(traits::get_accessor_type<F, 0>() == algorithm::detail::access_type::item);
            static_assert(std::is_invocable_v<F, cl::sycl::item<Rank>>, "generate kernels may only take cl::sycl::item<Rank> as input");
            const auto t = algorithm::detail::task<ExecutionPolicy>(generate_impl<ExecutionPolicy>(beg, end, f));
            return [=](distr_queue q) { t(q, beg, end); };
        }

        template <typename ExecutionPolicy, typename F, int Rank,
                  celerity::algorithm::require<celerity::algorithm::traits::arity_v<F> == 1> = celerity::algorithm::yes>
        auto generate(cl::sycl::range<Rank> range, const F &f)
        {
            //static_assert(traits::get_accessor_type<F, 0>() == access_type::item);
            static_assert(std::is_invocable_v<F, cl::sycl::item<Rank>>, "generate kernels may only take cl::sycl::item<Rank> as input");
            using value_type = std::invoke_result_t<F, cl::sycl::item<Rank>>;

            return package_generate<value_type>(
                [f](auto beg, auto end) { return task<ExecutionPolicy>(generate_impl<ExecutionPolicy>(beg, end, f)); },
                range);
        }

        // DISABLED
        template <typename ExecutionPolicy, typename F, typename T, int Rank,
                  celerity::algorithm::require<celerity::algorithm::traits::arity_v<F> == 0> = celerity::algorithm::yes>
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
        using execution_policy = algorithm::detail::named_distributed_execution_policy<KernelName>;
        return detail::generate<execution_policy>(range, f);
    }

    template <typename KernelName, typename T, int Rank, typename F>
    auto generate(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
    {
        return algorithm::generate(algorithm::distr<KernelName>(q), beg, end, f);
    }

    template <typename ExecutionPolicy, typename T, int Rank, typename F>
    auto generate(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
    {
        return generate(p, begin(in), end(in), f);
    }

    template <typename KernelName, typename T, int Rank, typename F>
    auto generate(celerity::distr_queue q, buffer<T, Rank> in, const F &f)
    {
        return hla::experimental::generate(algorithm::distr<KernelName>(q), begin(in), end(in), f);
    }

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_GENERATE_H