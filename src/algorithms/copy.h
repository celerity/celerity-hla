#ifndef COPY_H
#define COPY_H

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
template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	/*using policy_type = strip_queue_t<ExecutionPolicy>;

	static_assert(!policy_traits<std::decay_t<ExecutionPolicy>>::is_distributed);
	static_assert(!is_celerity_iterator_v<IteratorType>);

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, one_to_one>(cgh, beg, end);

		if constexpr (algorithm::is_contiguous_iterator<IteratorType>() &&
					  std::is_same_v<T, typename std::iterator_traits<IteratorType>::value_type>)
		{
			return [=]() { memcpy(out, in_acc.get_accessor().get_pointer(), distance(beg, end).size() * sizeof(T)); };
		}
		else
		{
			auto out_copy = out;
			return [=](auto item) { *out_copy++ = in_acc[item]; };
		}
	};*/
}

} // namespace detail

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	return task<ExecutionPolicy>(detail::copy(p, beg, end, out));
}

} // namespace actions

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	return scoped_sequence{actions::copy(p, beg, end, out), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer<T, Rank> in, IteratorType out)
{
	return copy(p, begin(in), end(in), out);
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer<T, Rank> in, buffer<T, Rank> out)
{
	return copy(p, begin(in), end(in), begin(out));
}

} // namespace celerity::algorithm

#endif // COPY_H