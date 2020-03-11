#ifndef FOR_EACH_H
#define FOR_EACH_H

#include "../iterator.h"
#include "../task.h"
#include "../task_sequence.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{
template <typename ExecutionPolicy, typename F, typename T, int Rank, template <typename, int> typename InIterator,
          typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(InIterator<T, Rank> beg, InIterator<T, Rank> end, const F &f)
{
    using policy_type = strip_queue_t<ExecutionPolicy>;
    using accessor_type = algorithm::detail::accessor_type_t<F, 1, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, accessor_type>(cgh, beg, end);

        return [=](item_context<Rank, T> &ctx) {
            f(ctx[0], in_acc[ctx[0]]);
        };
    };
}
} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto for_each(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return [=, t = task<ExecutionPolicy>(detail::for_each<ExecutionPolicy>(beg, end, f))](distr_queue q) {
        return t(q, beg, end);
    };
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return std::invoke(actions::for_each<ExecutionPolicy>(beg, end, f), p.q);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
    return for_each(p, begin(in), end(in), f);
}

} // namespace celerity::algorithm

#endif // FOR_EACH_H