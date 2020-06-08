#ifndef FOR_EACH_H
#define FOR_EACH_H

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
template <typename ExecutionPolicy, typename F, typename T, int Rank, template <typename, int> typename InIterator,
          require<traits::get_accessor_type<F, 0>() == access_type::item> = yes>
auto for_each_impl(InIterator<T, Rank> beg, InIterator<T, Rank> end, const F &f)
{
    using namespace traits;
    using namespace cl::sycl::access;

    using policy_type = strip_queue_t<ExecutionPolicy>;
    using accessor_type = accessor_type_t<F, 1, T>;

    return [=](celerity::handler &cgh) {
        auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, accessor_type>(cgh, beg, end);

        return [=](item_context<Rank, void(T)> &ctx) {
            f(ctx.get_item(), in_acc[ctx.get_in()]);
        };
    };
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto for_each(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    const auto t = task<ExecutionPolicy>(for_each_impl<ExecutionPolicy>(beg, end, f));
    return [=](distr_queue q) { t(q, beg, end); };
}

} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          require<traits::get_accessor_type<F, 0>() == detail::access_type::item> = yes>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return std::invoke(detail::for_each<ExecutionPolicy>(beg, end, f), p.q);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          require<traits::get_accessor_type<F, 0>() == detail::access_type::item> = yes>
auto for_each(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
    return for_each(p, begin(in), end(in), f);
}

} // namespace celerity::algorithm

#endif // FOR_EACH_H