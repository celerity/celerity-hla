#ifndef CELERITY_HLA_TRANSFORM_H
#define CELERITY_HLA_TRANSFORM_H

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
        template <typename ExecutionPolicy, template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename T, int Rank, Kernel<T> F>
        auto transform_impl(InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
        {
            using namespace cl::sycl::access;

            using policy_type = algorithm::traits::strip_queue_t<ExecutionPolicy>;

            return [=](celerity::handler &cgh) {
                auto in_acc = get_access<policy_type, mode::read, 0>(cgh, beg, end, f);
                auto out_acc = get_out_access<policy_type, mode::discard_write>(cgh, out, out);

                return [=](algorithm::detail::item_context<Rank, U(T)> &ctx) {
                    out_acc[ctx.get_out()] = f(in_acc[ctx.get_in()]);
                };
            };
        }

        // template <typename ExecutionPolicy, template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename F, typename T, int Rank,
        //           require<traits::function_traits<F>::arity == 2> = yes>
        // auto transform_impl(InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
        // {
        //     using namespace traits;
        //     using namespace cl::sycl::access;

        //     using policy_type = strip_queue_t<ExecutionPolicy>;
        //     using accessor_type = accessor_type_t<F, 1, T>;

        //     return [=](celerity::handler &cgh) {
        //         auto in_acc = get_access<policy_type, mode::read, accessor_type>(cgh, beg, end);
        //         auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

        //         return [=](item_context<Rank, U(T)> &ctx) {
        //             out_acc[ctx.get_out()] = f(ctx.get_item(), in_acc[ctx.get_in()]);
        //         };
        //     };
        // }

        template <typename ExecutionPolicy, typename T, typename U, int Rank, Kernel<T> F>
        auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
        {
            const auto t = algorithm::detail::task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, out, f));
            return [=](distr_queue q) { t(q, beg, end); };
        }

        template <typename ExecutionPolicy, typename F>
        auto transform(const F &f)
        {
            return package_transform<F>(
                [f](auto beg, auto end, auto out) { return algorithm::detail::task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, out, f)); });
        }

        // template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
        //           require<traits::arity_v<F> == 2,
        //                   traits::get_accessor_type<F, 0>() == access_type::item> = yes>
        // auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
        // {
        //     const auto t = task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, out, f));
        //     return [=](distr_queue q) { t(q, beg, end); };
        // }

        // template <typename ExecutionPolicy, typename F,
        //           require<traits::arity_v<F> == 2,
        //                   traits::get_accessor_type<F, 0>() == access_type::item> = yes>
        // auto transform(const F &f)
        // {
        //     constexpr auto access_type = traits::get_accessor_type<F, 1>();

        //     return package_transform<access_type, F>(
        //         [f](auto beg, auto end, auto out) { return task<ExecutionPolicy>(transform_impl<ExecutionPolicy>(beg, end, out, f)); });
        // }

    } // namespace detail

    template <typename ExecutionPolicy, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
    {
        return std::invoke(detail::transform<ExecutionPolicy>(beg, end, out, f), p.q);
    }

    template <typename KernelName, typename F>
    auto transform(const F &f)
    {
        using execution_policy = algorithm::detail::named_distributed_execution_policy<KernelName>;
        return detail::transform<execution_policy>(f);
    }

    template <typename KernelName, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
    {
        return hla::experimental::transform(algorithm::distr<KernelName>(q), beg, end, out, f);
    }

    template <typename ExecutionPolicy, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
    {
        return transform(p, begin(in), end(in), out, f);
    }

    template <typename KernelName, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
    {
        return transform(algorithm::distr<KernelName>(q), in, out, f);
    }

    template <typename ExecutionPolicy, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
    {
        return transform(p, begin(in), end(in), begin(out), f);
    }

    template <typename KernelName, typename T, typename U, int Rank, Kernel<T> F>
    auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
    {
        return transform(algorithm::distr<KernelName>(q), in, out, f);
    }

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_TRANSFORM_H
