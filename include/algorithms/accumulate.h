#ifndef ACCUMULATE_H
#define ACCUMULATE_H

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
template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank,
		  require<traits::get_accessor_type<BinaryOp, 1>() == access_type::one_to_one> = yes>
auto accumulate_impl(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	/*using policy_type = strip_queue_t<ExecutionPolicy>;

	static_assert(!policy_traits<ExecutionPolicy>::is_distributed, "can not be distributed");
	static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

	return [=](celerity::handler &cgh) {
		const auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, one_to_one>(cgh, beg, end);

		auto sum = init;

		dispatch<policy_type>(cgh, beg, end, [&](auto item) {
			sum = op(std::move(sum), in_acc[item]);
		});

		return sum;
	};*/
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return task<ExecutionPolicy>(accumulate_impl(p, beg, end, init, op));
}

} // namespace detail

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return std::invoke(detail::accumulate(p, beg, end, init, op), p.q);
}

template <typename KernelName, typename BinaryOp, typename T, int Rank>
auto accumulate(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return accumulate(distr<KernelName>(q), beg, end, init, op);
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer<T, Rank> in, T init, const BinaryOp &op)
{
	return accumulate(p, begin(in), end(in), init, op);
}

template <typename KernelName, typename BinaryOp, typename T, int Rank>
auto accumulate(celerity::distr_queue q, buffer<T, Rank> in, T init, const BinaryOp &op)
{
	return accumulate(distr<KernelName>(q), begin(in), end(in), init, op);
}

} // namespace celerity::algorithm

#endif // ACCUMULATE_H