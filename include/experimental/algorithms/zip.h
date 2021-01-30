#ifndef CELERITY_HLA_ZIP_H
#define CELERITY_HLA_ZIP_H

#include "../../iterator.h"
#include "../../task.h"
#include "../../policy.h"
#include "../../require.h"

#include "../../experimental/packaged_tasks/packaged_transform.h"
#include "../../experimental/accessor_proxies.h"

using celerity::algorithm::buffer_iterator;

namespace celerity::hla::experimental
{
    namespace detail
    {
        template <typename ExecutionPolicy,
                  template <typename, int> typename FirstInputIteratorType,
                  template <typename, int> typename SecondInputIteratorType,
                  template <typename, int> typename OutputIteratorType,
                  typename T,
                  typename U,
                  typename V,
                  Kernel<T, U> F,
                  int Rank>
        auto zip_impl(FirstInputIteratorType<T, Rank> beg,
                      FirstInputIteratorType<T, Rank> end,
                      SecondInputIteratorType<U, Rank> beg2,
                      OutputIteratorType<V, Rank> out,
                      const F &f)
        {
            using namespace cl::sycl::access;

            using policy_type = algorithm::traits::strip_queue_t<ExecutionPolicy>;

            return [=](celerity::handler &cgh) {
                auto first_in_acc = get_access<policy_type, mode::read, 0, 2>(cgh, beg, end, f);
                auto second_in_acc = get_access<policy_type, mode::read, 1, 2>(cgh, beg2, beg2, f);
                auto out_acc = get_out_access<policy_type, mode::discard_write>(cgh, out, out);

                return [=](algorithm::detail::item_context<Rank, V(T, U)> &ctx) {
                    out_acc[ctx.get_out()] = f(first_in_acc[ctx.template get_in<0>()], second_in_acc[ctx.template get_in<1>()]);
                };
            };
        }

        // template <typename ExecutionPolicy,
        //           template <typename, int> typename FirstInputIteratorType,
        //           template <typename, int> typename SecondInputIteratorType,
        //           template <typename, int> typename OutputIteratorType,
        //           typename F,
        //           typename T,
        //           typename U,
        //           typename V,
        //           int Rank,
        //           require<traits::function_traits<F>::arity == 3> = yes>
        // auto transform_impl(FirstInputIteratorType<T, Rank> beg,
        //                     FirstInputIteratorType<T, Rank> end,
        //                     SecondInputIteratorType<U, Rank> beg2,
        //                     OutputIteratorType<V, Rank> out, const F &f)
        // {
        //     using namespace traits;
        //     using namespace cl::sycl::access;

        //     using policy_type = strip_queue_t<ExecutionPolicy>;
        //     using first_accessor_type = accessor_type_t<F, 1, T>;
        //     using second_accessor_type = accessor_type_t<F, 2, T>;

        //     return [=](celerity::handler &cgh) {
        //         auto first_in_acc = get_access<policy_type, mode::read, first_accessor_type>(cgh, beg, end);
        //         auto second_in_acc = get_access<policy_type, mode::read, second_accessor_type>(cgh, beg2, beg2);
        //         auto out_acc = get_access<policy_type, mode::discard_write, one_to_one>(cgh, out, out);

        //         return [=](item_context<Rank, V(T, U)> &ctx) {
        //             out_acc[ctx.get_out()] = f(ctx.get_item(), first_in_acc[ctx.template get_in<0>()], second_in_acc[ctx.template get_in<1>()]);
        //         };
        //     };
        // }

        template <typename ExecutionPolicy, typename T, typename U, int Rank, Kernel<T, U> F>
        auto zip(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
        {
            const auto t = algorithm::detail::task<ExecutionPolicy>(zip_impl<ExecutionPolicy>(beg, end, beg2, out, f));
            return [=](distr_queue q) { t(q, beg, end); };
        }

        template <typename ExecutionPolicy, typename F>
        auto zip(const F &f)
        {
            return package_zip<F>([f](auto beg, auto end, auto beg2, auto out) {
                return task<ExecutionPolicy>(zip_impl<ExecutionPolicy>(beg, end, beg2, out, f));
            });
        }

        // template <typename ExecutionPolicy, typename T, int Rank, typename F, typename U,
        //           require<traits::arity_v<F> == 3> = yes>
        // auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
        // {
        //     const auto t = task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, beg2, out, f));
        //     return [=](distr_queue q) { t(q, beg, end); };
        // }

        // template <typename ExecutionPolicy, typename F,
        //           require<traits::arity_v<F> == 3> = yes>
        // auto transform(const F &f)
        // {
        //     constexpr auto first_access_type = traits::get_accessor_type<F, 1>();
        //     constexpr auto second_access_type = traits::get_accessor_type<F, 2>();

        //     return package_zip<first_access_type, second_access_type, F>(
        //         [f](auto beg, auto end, auto beg2, auto out) { return task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, beg2, out, f)); });
        // }

    } // namespace detail

    template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
    {
        return std::invoke(detail::zip<ExecutionPolicy>(beg, end, beg2, out, f), p.q);
    }

    template <typename KernelName, typename F>
    auto zip(const F &f)
    {
        using execution_policy = algorithm::detail::named_distributed_execution_policy<KernelName>;
        return detail::zip<execution_policy>(f);
    }

    template <typename KernelName, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
    {
        return zip(algorithm::distr<KernelName>(q), beg, end, beg2, out, f);
    }

    template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
    {
        return zip(p, begin(in), end(in), beg2, out, f);
    }

    template <typename KernelName, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
    {
        return zip(algorithm::distr<KernelName>(q), begin(in), end(in), beg2, out, f);
    }

    template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
    {
        return zip(p, begin(first), end(first), begin(second), out, f);
    }

    template <typename KernelName, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
    {
        return zip(algorithm::distr<KernelName>(q), begin(first), end(first), begin(second), out, f);
    }

    template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
    {
        return zip(p, begin(first), end(first), begin(second), begin(out), f);
    }

    template <typename KernelName, typename T, typename U, typename V, int Rank, Kernel<T, V> F>
    auto zip(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
    {
        return zip(algorithm::distr<KernelName>(q), begin(first), end(first), begin(second), begin(out), f);
    }

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_ZIP_H
