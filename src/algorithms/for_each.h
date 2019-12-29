#ifndef FOR_EACH_H
#define FOR_EACH_H

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
template <typename InputAccessorType, typename ExecutionPolicy, typename F, typename T, int Rank,
          typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    /*using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, InputAccessorType>(cgh, beg, end);

		dispatch<policy_type>(cgh, beg, end, [=](auto item) {
			f(item, in_acc[item]);
		});
	};*/
}
} // namespace detail

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return task<ExecutionPolicy>(detail::for_each<algorithm::detail::accessor_type_t<F, 1, T>>(p, beg, end, f));
}

} // namespace actions

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
    return scoped_sequence{actions::for_each(p, beg, end, f), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T, int Rank, typename F,
          typename = std::enable_if_t<detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
    return for_each(p, begin(in), end(in), f);
}

} // namespace celerity::algorithm

#endif // FOR_EACH_H