#ifndef COPY_H
#define COPY_H

#include "../iterator.h"
#include "../task.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"

namespace celerity::algorithm
{
namespace detail
{
template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy_impl(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	using namespace traits;
	using namespace cl::sycl::access;

	using policy_type = strip_queue_t<ExecutionPolicy>;

	static_assert(!policy_traits<std::decay_t<ExecutionPolicy>>::is_distributed);
	static_assert(!is_celerity_iterator_v<IteratorType>);

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, one_to_one>(cgh, beg, end);

		if constexpr (traits::is_contiguous_iterator<IteratorType>() &&
					  sizeof(T) == sizeof(typename std::iterator_traits<IteratorType>::value_type))
		{
			return [=]() { memcpy(out, in_acc.get_accessor().get_pointer(), distance(beg, end).size() * sizeof(T)); };
		}
		else
		{
			static_assert(std::is_void_v<T>, "not supported");
			//return [=](auto item) mutable { *out_copy++ = in_acc[item]; };
		}
	};
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, int)
{
	return task<ExecutionPolicy>(copy_impl(p, beg, end, out));
}

} // namespace detail

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	return std::invoke(detail::copy(p, beg, end, out, 0), p.q);
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

template <typename T, int Rank>
auto make_buffer(T *data, cl::sycl::range<Rank> range) -> celerity::buffer<T, Rank>
{
	return {data, range};
}

} // namespace celerity::algorithm

#endif // COPY_H