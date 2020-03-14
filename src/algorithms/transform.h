#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "../iterator.h"
#include "../task.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"
#include "../require.h"

namespace celerity::algorithm
{

namespace actions
{
namespace detail
{

template <typename ExecutionPolicy, template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename F, typename T, int Rank,
          require<algorithm::detail::function_traits<F>::arity == 1> = yes>
auto transform(InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
{
    using namespace cl::sycl::access;

    using policy_type = strip_queue_t<ExecutionPolicy>;
    using accessor_type = algorithm::detail::accessor_type_t<F, 0, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, mode::read, accessor_type>(cgh, beg, end);
        auto out_acc = get_access<policy_type, mode::discard_write, one_to_one>(cgh, out, out);

        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = f(in_acc[ctx[0]]);
        };
    };
}

template <typename ExecutionPolicy, template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename F, typename T, int Rank,
          require<algorithm::detail::function_traits<F>::arity == 2> = yes>
auto transform(InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    using accessor_type = algorithm::detail::accessor_type_t<F, 1, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, mode::read, accessor_type>(cgh, beg, end);
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = f(ctx.get_item(), in_acc[ctx[0]]);
        };
    };
}

template <typename ExecutionPolicy,
          template <typename, int> typename FirstInputIteratorType,
          template <typename, int> typename SecondInputIteratorType,
          template <typename, int> typename OutputIteratorType,
          typename F,
          typename T,
          typename U,
          int Rank,
          require<algorithm::detail::function_traits<F>::arity == 2> = yes>
auto transform(FirstInputIteratorType<T, Rank> beg,
               FirstInputIteratorType<T, Rank> end,
               SecondInputIteratorType<U, Rank> beg2,
               OutputIteratorType<T, Rank> out,
               const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    using first_accessor_type = algorithm::detail::accessor_type_t<F, 0, T>;
    using second_accessor_type = algorithm::detail::accessor_type_t<F, 1, U>;

    return [=](celerity::handler &cgh) {
        auto first_in_acc = get_access<policy_type, mode::read, first_accessor_type>(cgh, beg, end);
        auto second_in_acc = get_access<policy_type, mode::read, second_accessor_type>(cgh, beg2, beg2);
        auto out_acc = get_access<policy_type, mode::discard_write, one_to_one>(cgh, out, out);

        // TODO: item_context needs to fit for both T and U
        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = f(first_in_acc[ctx[0]], second_in_acc[ctx[1]]);
        };
    };
}

template <typename ExecutionPolicy,
          template <typename, int> typename FirstInputIteratorType,
          template <typename, int> typename SecondInputIteratorType,
          template <typename, int> typename OutputIteratorType,
          typename F,
          typename T,
          typename U,
          int Rank,
          require<algorithm::detail::function_traits<F>::arity == 3> = yes>
auto transform(FirstInputIteratorType<T, Rank> beg,
               FirstInputIteratorType<T, Rank> end,
               SecondInputIteratorType<U, Rank> beg2,
               OutputIteratorType<T, Rank> out, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    using first_accessor_type = algorithm::detail::accessor_type_t<F, 1, T>;
    using second_accessor_type = algorithm::detail::accessor_type_t<F, 2, T>;

    return [=](celerity::handler &cgh) {
        auto first_in_acc = get_access<policy_type, mode::read, first_accessor_type>(cgh, beg, end);
        auto second_in_acc = get_access<policy_type, mode::read, second_accessor_type>(cgh, beg2, beg2);
        auto out_acc = get_access<policy_type, mode::discard_write, one_to_one>(cgh, out, out);

        // TODO: item_context needs to fit for both T and U
        return [=](item_context<Rank, T> &ctx) {
            out_acc[ctx[0]] = f(ctx.get_item(), first_in_acc[ctx[0]], second_in_acc[ctx[1]]);
        };
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<algorithm::detail::arity_v<F> == 1,
                  algorithm::detail::get_accessor_type<F, 0>() != access_type::item> = yes>
auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    const auto t = task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, out, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F,
          require<algorithm::detail::arity_v<F> == 1,
                  algorithm::detail::get_accessor_type<F, 0>() != access_type::item> = yes>
auto transform(const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 0>();

    return package_transform<access_type, F>(
        [f](auto beg, auto end, auto out) { return task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, out, f)); });
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<algorithm::detail::arity_v<F> == 2,
                  algorithm::detail::get_accessor_type<F, 0>() == access_type::item> = yes>
auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    const auto t = task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, out, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F,
          require<algorithm::detail::arity_v<F> == 2,
                  algorithm::detail::get_accessor_type<F, 0>() == access_type::item> = yes>
auto transform(const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 1>();

    return package_transform<access_type, F>(
        [f](auto beg, auto end, auto out) { return task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, out, f)); });
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<algorithm::detail::arity_v<F> == 2,
                  algorithm::detail::get_accessor_type<F, 0>() != access_type::item> = yes>
auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    const auto t = task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, beg2, out, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F,
          require<algorithm::detail::arity_v<F> == 2,
                  algorithm::detail::get_accessor_type<F, 0>() != access_type::item> = yes>
auto transform(const F &f)
{
    constexpr auto first_access_type = algorithm::detail::get_accessor_type<F, 0>();
    constexpr auto second_access_type = algorithm::detail::get_accessor_type<F, 1>();

    return package_zip<first_access_type, second_access_type, F>(
        [f](auto beg, auto end, auto beg2, auto out) { return task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, beg2, out, f)); });
}

template <typename ExecutionPolicy, typename T, int Rank, typename F, typename U,
          require<algorithm::detail::arity_v<F> == 3> = yes>
auto transform(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    const auto t = task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, beg2, out, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

template <typename ExecutionPolicy, typename F,
          require<algorithm::detail::arity_v<F> == 3> = yes>
auto transform(const F &f)
{
    constexpr auto first_access_type = algorithm::detail::get_accessor_type<F, 1>();
    constexpr auto second_access_type = algorithm::detail::get_accessor_type<F, 2>();

    return package_zip<first_access_type, second_access_type, F>(
        [f](auto beg, auto end, auto beg2, auto out) { return task<ExecutionPolicy>(detail::transform<ExecutionPolicy>(beg, end, beg2, out, f)); });
}

} // namespace actions

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    return std::invoke(actions::transform<ExecutionPolicy>(beg, end, out, f), p.q);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return std::invoke(actions::transform<ExecutionPolicy>(beg, end, beg2, out, f), p.q);
}

template <typename KernelName, typename F>
auto transform(const F &f)
{
    using execution_policy = named_distributed_execution_policy<KernelName>;
    return actions::transform<execution_policy>(f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), beg, end, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), out, f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), in, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), begin(out), f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), in, out, f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), beg, end, beg2, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), beg2, out, f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(in), end(in), beg2, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(first), end(first), begin(second), out, f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(first), end(first), begin(second), out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
{
    return transform(p, begin(first), end(first), begin(second), begin(out), f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          require<detail::get_accessor_type<F, 0>() != access_type::invalid,
                  detail::get_accessor_type<F, 1>() != access_type::invalid> = yes>
auto transform(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(first), end(first), begin(second), begin(out), f);
}

} // namespace celerity::algorithm

#endif // TRANSFORM_H
