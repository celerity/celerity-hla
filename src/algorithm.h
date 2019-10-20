#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "iterator.h"
#include "task.h"
#include "task_sequence.h"
#include "accessor_proxy.h"
#include "policy.h"
#include "scoped_sequence.h"
#include "decorated_task.h"

#include <future>

namespace celerity::algorithm
{
namespace actions
{
namespace detail
{
template <typename InputAccessorType, typename U, typename ExecutionPolicy, typename F, typename T, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, mode::read, InputAccessorType>(cgh, beg, end);
		auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

		return [=](auto item) { out_acc[item] = f(in_acc[item]); };
	};
}

template <typename InputAccessorType, typename U, typename ExecutionPolicy, typename F, typename T, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, mode::read, InputAccessorType>(cgh, beg, end);
		auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, out, out);

		return [=](auto item) { out_acc[item] = f(item, in_acc[item]); };
	};
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	assert(beg.get_buffer().get_id() == end.get_buffer().get_id());
	assert(beg.get_buffer().get_id() != beg2.get_buffer().get_id());
	assert(beg.get_buffer().get_id() != out.get_buffer().get_id());

	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto first_in_acc = get_access<policy_type, mode::read, FirstInputAccessorType>(cgh, beg, end);
		auto second_in_acc = get_access<policy_type, mode::read, SecondInputAccessorType>(cgh, beg2, beg2);
		auto out_acc = get_access<policy_type, mode::write, OutputAccessorType>(cgh, out, out);

		return [=](auto item) { out_acc[item] = f(first_in_acc[item], second_in_acc[item]); };
	};
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 3, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	assert(beg.get_buffer().get_id() == end.get_buffer().get_id());
	assert(beg.get_buffer().get_id() != beg2.get_buffer().get_id());
	assert(beg.get_buffer().get_id() != out.get_buffer().get_id());

	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto first_in_acc = get_access<policy_type, mode::read, FirstInputAccessorType>(cgh, beg, end);
		auto second_in_acc = get_access<policy_type, mode::read, SecondInputAccessorType>(cgh, beg2, beg2);
		auto out_acc = get_access<policy_type, mode::write, OutputAccessorType>(cgh, out, out);

		return [=](auto item) { out_acc[item] = f(item, first_in_acc[item], second_in_acc[item]); };
	};
}

template <typename ExecutionPolicy, typename F, typename T, int Rank>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	assert(beg.get_buffer().get_id() == end.get_buffer().get_id());

	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);

		if constexpr (algorithm::detail::get_accessor_type<F, 0>() == access_type::item)
		{
			return [=](auto item) { out_acc[item] = f(item); };
		}
		else
		{
			return [=](auto item) { out_acc[item] = f(); };
		}
	};
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	assert(beg.get_buffer().get_id() == end.get_buffer().get_id());

	using policy_type = strip_queue_t<ExecutionPolicy>;
	using namespace cl::sycl::access;

	return [=](celerity::handler &cgh) {
		auto out_acc = get_access<policy_type, mode::write, one_to_one>(cgh, beg, end);
		return [=](auto item) { out_acc[item] = value; };
	};
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank, typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<BinaryOp, 1>() == access_type::one_to_one>>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
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

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
	return decorate_transform<algorithm::detail::get_accessor_type<F, 0>()>(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>>(p, beg, end, out, f)),
		beg, end, out);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
	return decorate_transform<algorithm::detail::get_accessor_type<F, 0>()>(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 1, T>>(p, beg, end, out, f)),
		beg, end, out);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	return decorated_task(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>, algorithm::detail::accessor_type_t<F, 1, U>, one_to_one>(p, beg, end, beg2, out, f)),
		beg, end);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F, typename U,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 3, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	return decorated_task(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 1, T>, algorithm::detail::accessor_type_t<F, 2, U>, one_to_one>(p, beg, end, beg2, out, f)),
		beg, end);
}

template <typename ExecutionPolicy, typename F, typename T, int Rank,
		  typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::item>>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	return task<ExecutionPolicy>(detail::generate(p, beg, end, f));
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	return decorate_generate<T>(task<ExecutionPolicy>(detail::fill(p, beg, end, value)), beg, end);
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return task<ExecutionPolicy>(detail::accumulate(p, beg, end, init, op));
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	return task<ExecutionPolicy>(detail::for_each<algorithm::detail::accessor_type_t<F, 1, T>>(p, beg, end, f));
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	return task<ExecutionPolicy>(detail::copy(p, beg, end, out));
}

template <typename F>
auto master_task(const F &f)
{
	static_assert(algorithm::detail::_is_master_task_v<F>, "not a compute task");
	return task<non_blocking_master_execution_policy>(f);
}

} // namespace actions

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> out, const F &f)
{
	return scoped_sequence{actions::transform(p, beg, end, out, f), submit_to(p.q)};
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

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	return scoped_sequence{actions::generate(p, beg, end, f), submit_to(p.q)};
}

template <typename KernelName, typename T, int Rank, typename F>
auto generate(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	return generate(distr<KernelName>(q), beg, end, f);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
	return generate(p, begin(in), end(in), f);
}

template <typename KernelName, typename T, int Rank, typename F>
auto generate(celerity::distr_queue q, buffer<T, Rank> in, const F &f)
{
	return generate(distr<KernelName>(q), begin(in), end(in), f);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	return scoped_sequence{actions::fill(p, beg, end, value), submit_to(p.q)};
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	return fill(distr<KernelName>(q), beg, end, value);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer<T, Rank> in, const T &value)
{
	return fill(p, begin(in), end(in), value);
}

template <typename KernelName, typename T, int Rank>
auto fill(celerity::distr_queue q, buffer<T, Rank> in, const T &value)
{
	return fill(distr<KernelName>(q), begin(in), end(in), value);
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return scoped_sequence{actions::accumulate(p, beg, end, init, op), submit_to(p.q)};
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

template <typename ExecutionPolicy, typename F>
auto master_task(ExecutionPolicy p, const F &f)
{
	static_assert(std::is_same_v<non_blocking_master_execution_policy, std::decay_t<ExecutionPolicy>>, "non-blocking master only");
	return scoped_sequence{actions::master_task(f), submit_to(p.q)};
}

} // namespace celerity::algorithm

#endif
