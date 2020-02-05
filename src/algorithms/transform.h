#ifndef TRANSFORM_H
#define TRANSFORM_H

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

template <template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename ExecutionPolicy, typename F, typename T, int Rank,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
{
    using namespace cl::sycl::access;

    using policy_type = strip_queue_t<ExecutionPolicy>;
    using accessor_type = algorithm::detail::accessor_type_t<F, 0, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, mode::read, accessor_type>(cgh, beg, end);
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

        return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(in_acc[item]); };
    };
}

template <template <typename, int> typename InIterator, template <typename, int> typename OutIterator, typename U, typename ExecutionPolicy, typename F, typename T, int Rank,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, InIterator<T, Rank> beg, InIterator<T, Rank> end, OutIterator<U, Rank> out, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    using accessor_type = algorithm::detail::accessor_type_t<F, 1, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, mode::read, accessor_type>(cgh, beg, end);
        auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

        return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(item, in_acc[item]); };
    };
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto first_in_acc = get_access<policy_type, mode::read, FirstInputAccessorType>(cgh, beg, end);
        auto second_in_acc = get_access<policy_type, mode::read, SecondInputAccessorType>(cgh, beg2, beg2);
        auto out_acc = get_access<policy_type, mode::write, OutputAccessorType>(cgh, out, out);

        return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(first_in_acc[item], second_in_acc[item]); };
    };
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 3, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using namespace cl::sycl::access;

    return [=](celerity::handler &cgh) {
        auto first_in_acc = get_access<policy_type, mode::read, FirstInputAccessorType>(cgh, beg, end);
        auto second_in_acc = get_access<policy_type, mode::read, SecondInputAccessorType>(cgh, beg2, beg2);
        auto out_acc = get_access<policy_type, mode::write, OutputAccessorType>(cgh, out, out);

        return [=](cl::sycl::item<Rank> item) { out_acc[item] = f(item, first_in_acc[item], second_in_acc[item]); };
    };
}

} // namespace detail

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 0>();

    return package_transform<access_type>(
        [p, f](auto _beg, auto _end, auto _out) { return task<ExecutionPolicy>(detail::transform(p, _beg, _end, _out, f)); },
        beg, end, out);
}

template <typename ExecutionPolicy, typename F,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1 && algorithm::detail::get_accessor_type<F, 0>() != access_type::item, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator_placeholder, buffer_iterator_placeholder, buffer_iterator_placeholder, const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 0>();

    return package_transform<access_type, F>(
        [p, f](auto beg, auto end, auto out) { return task<ExecutionPolicy>(detail::transform(p, beg, end, out, f)); });
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 1>();

    return package_transform<access_type>(
        [p, f](auto _beg, auto _end, auto _out) { return task<ExecutionPolicy>(detail::transform(p, _beg, _end, _out, f)); },
        beg, end, out);
}

template <typename ExecutionPolicy, typename F,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator_placeholder, buffer_iterator_placeholder, buffer_iterator_placeholder, const F &f)
{
    constexpr auto access_type = algorithm::detail::get_accessor_type<F, 1>();

    return package_transform<access_type, F>(
        [p, f](auto beg, auto end, auto out) { return task<ExecutionPolicy>(detail::transform(p, beg, end, out, f)); });
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    return package_zip<algorithm::detail::get_accessor_type<F, 0>(), algorithm::detail::get_accessor_type<F, 1>()>(
        task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>, algorithm::detail::accessor_type_t<F, 1, U>, one_to_one>(p, beg, end, beg2, out, f)),
        beg, end, beg2, out);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F, typename U,
          ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 3, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
    return package_zip<algorithm::detail::get_accessor_type<F, 1>(), algorithm::detail::get_accessor_type<F, 2>()>(
        task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 1, T>, algorithm::detail::accessor_type_t<F, 2, U>, one_to_one>(p, beg, end, beg2, out, f)),
        beg, end, beg2, out);
}

} // namespace actions

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    return scoped_sequence{actions::transform(p, beg, end, out, f), submit_to(p.q)};
}

template <typename KernelName, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer_iterator_placeholder, buffer_iterator_placeholder, buffer_iterator_placeholder, const F &f)
{
    return actions::transform(distr<KernelName>(q), {}, {}, {}, f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), beg, end, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), out, f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(in), end(in), out, f);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), begin(out), f);
}

template <typename KernelName, typename T, typename U, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(in), end(in), begin(out), f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return scoped_sequence{actions::transform(p, beg, end, beg2, out, f), submit_to(p.q)};
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), beg, end, beg2, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(in), end(in), beg2, out, f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(in), end(in), beg2, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(p, begin(first), end(first), begin(second), out, f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(first), end(first), begin(second), out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
{
    return transform(p, begin(first), end(first), begin(second), begin(out), f);
}

template <typename KernelName, typename T, typename U, typename V, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
                                      detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(celerity::distr_queue q, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
{
    return transform(distr<KernelName>(q), begin(first), end(first), begin(second), begin(out), f);
}

} // namespace celerity::algorithm

#endif // TRANSFORM_H
