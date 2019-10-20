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
template <typename ExecutionPolicy, typename Iterator, typename F>
void dispatch(celerity::handler &cgh, Iterator beg, Iterator end, const F f)
{
	using execution_policy_type = std::decay_t<ExecutionPolicy>;

	const auto r = distance(beg, end);

	if constexpr (policy_traits<execution_policy_type>::is_distributed)
	{
		cgh.template parallel_for<typename policy_traits<execution_policy_type>::kernel_name>(r, *beg, f);
	}
	else
	{
		cgh.run([&]() { for_each_index(beg, end, r, *beg, f); });
	}
}

template <typename InputAccessorType, typename IteratorType, typename ExecutionPolicy, typename F, typename T, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, InputAccessorType>(cgh, beg, end);

		if constexpr (is_celerity_iterator_v<IteratorType>)
		{
			auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, one_to_one>(cgh, out, out);
			return [=](auto item) { out_acc[item] = f(in_acc[item]); };
		}
		else
		{
			auto out_tmp = out;
			return [=](auto item) { *out_tmp++ = f(in_acc[item]); };
		}
	};
}

template <typename InputAccessorType, typename IteratorType, typename ExecutionPolicy, typename F, typename T, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, InputAccessorType>(cgh, beg, end);

		if constexpr (is_celerity_iterator_v<IteratorType>)
		{
			auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, one_to_one>(cgh, out, out);
			return [=](auto item) { out_acc[item] = f(item, in_acc[item]); };
		}
		else
		{
			auto out_tmp = out;
			return [=](auto item) { *out_tmp++ = f(item, in_acc[item]); };
		}
	};
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		/*if (beg.get_buffer().get_id() == out.get_buffer().get_id())
		{
			auto in_out_acc = get_access<policy_type, cl::sycl::access::mode::read_write, FirstInputAccessorType>(cgh, beg, end);
			auto second_in_acc = get_access<policy_type, cl::sycl::access::mode::read, SecondInputAccessorType>(cgh, beg2, beg2);

			dispatch(p, cgh, beg, end, [&](auto item) {
				in_out_acc[item] = f(in_out_acc[item], second_in_acc[item]);
			});
		}
		else*/
		{
			auto first_in_acc = get_access<policy_type, cl::sycl::access::mode::read, FirstInputAccessorType>(cgh, beg, end);
			auto second_in_acc = get_access<policy_type, cl::sycl::access::mode::read, SecondInputAccessorType>(cgh, beg2, beg2);

			auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, OutputAccessorType>(cgh, out, out);

			return [=](auto item) { out_acc[item] = f(first_in_acc[item], second_in_acc[item]); };
		}
	};
}

template <typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, typename U, int Rank,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 3, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<U, Rank> beg2, buffer_iterator<T, Rank> out, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		/*if (beg.get_buffer().get_id() == out.get_buffer().get_id())
		{
			auto in_out_acc = get_access<policy_type, cl::sycl::access::mode::read_write, FirstInputAccessorType>(cgh, beg, end);
			auto second_in_acc = get_access<policy_type, cl::sycl::access::mode::read, SecondInputAccessorType>(cgh, beg2, beg2);

			dispatch<policy_type>(cgh, beg, end, [=](auto item) {
				in_out_acc[item] = f(item, in_out_acc[item], second_in_acc[item]);
			});
		}
		else*/
		{
			auto first_in_acc = get_access<policy_type, cl::sycl::access::mode::read, FirstInputAccessorType>(cgh, beg, end);
			auto second_in_acc = get_access<policy_type, cl::sycl::access::mode::read, SecondInputAccessorType>(cgh, beg2, beg2);

			auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, OutputAccessorType>(cgh, out, out);

			dispatch<policy_type>(cgh, beg, end, [=](auto item) {
				out_acc[item] = f(item, first_in_acc[item], second_in_acc[item]);
			});
		}
	};
}

template <typename ExecutionPolicy, typename F, typename T, int Rank>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, one_to_one>(cgh, beg, end);

		dispatch<policy_type>(cgh, beg, end, [=](auto item) {
			if constexpr (algorithm::detail::get_accessor_type<F, 0>() == access_type::item)
			{
				out_acc[item] = f(item);
			}
			else
			{
				out_acc[item] = f();
			}
		});
	};
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto out_acc = get_access<policy_type, cl::sycl::access::mode::write, one_to_one>(cgh, beg, end);
		return [=](auto item) { out_acc[item] = value; };
	};
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank, typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<BinaryOp, 1>() == access_type::one_to_one>>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	static_assert(!policy_traits<ExecutionPolicy>::is_distributed, "can not be distributed");
	static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

	return [=](celerity::handler &cgh) {
		const auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, one_to_one>(cgh, beg, end);

		auto sum = init;

		dispatch<policy_type>(cgh, beg, end, [&](auto item) {
			sum = op(std::move(sum), in_acc[item]);
		});

		return sum;
	};
}

template <typename InputAccessorType, typename ExecutionPolicy, typename F, typename T, int Rank,
		  typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::item>>
auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, InputAccessorType>(cgh, beg, end);

		dispatch<policy_type>(cgh, beg, end, [=](auto item) {
			f(item, in_acc[item]);
		});
	};
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank>
auto copy(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;

	static_assert(!policy_traits<std::decay_t<ExecutionPolicy>>::is_distributed);
	static_assert(!is_celerity_iterator_v<IteratorType>);

	return [=](celerity::handler &cgh) {
		auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, one_to_one>(cgh, beg, end);

		if constexpr (algorithm::is_contiguous_iterator<IteratorType>() &&
					  std::is_same_v<T, typename std::iterator_traits<IteratorType>::value_type>)
		{
			memcpy(out, in_acc.get_accessor().get_pointer(), distance(beg, end).size() * sizeof(T));
		}
		else
		{
			auto out_copy = out;
			dispatch<policy_type>(cgh, beg, end, [&](auto item) {
				*out_copy++ = in_acc[item];
			});
		}
	};
}
} // namespace detail

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank, typename F,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 1, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, const F &f)
{
	return decorated_task(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>>(p, beg, end, out, f)),
		beg, end);
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank, typename F,
		  ::std::enable_if_t<algorithm::detail::function_traits<F>::arity == 2, int> = 0>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, const F &f)
{
	return decorated_task(
		task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 1, T>>(p, beg, end, out, f)),
		beg, end);
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
	return task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 1, T>, algorithm::detail::accessor_type_t<F, 2, U>, one_to_one>(p, beg, end, beg2, out, f));
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
	return decorated_task(task<ExecutionPolicy>(detail::fill(p, beg, end, value)), beg, end);
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

} // namespace actions

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, IteratorType out, const F &f)
{
	return scoped_sequence{actions::transform(p, beg, end, out, f), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename IteratorType, typename T, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, IteratorType out, const F &f)
{
	return transform(p, begin(in), end(in), out, f);
}

template <typename ExecutionPolicy, typename T, typename U, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer<U, Rank> out, const F &f)
{
	return transform(p, begin(in), end(in), begin(out), f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
									  detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
	return scoped_sequence{actions::transform(p, beg, end, beg2, out, f), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
									  detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> in, buffer_iterator<V, Rank> beg2, buffer_iterator<U, Rank> out, const F &f)
{
	return transform(p, begin(in), end(in), beg2, out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
									  detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer_iterator<U, Rank> out, const F &f)
{
	return transform(p, begin(first), end(first), begin(second), out, f);
}

template <typename ExecutionPolicy, typename T, typename U, typename V, int Rank, typename F,
		  typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
									  detail::get_accessor_type<F, 1>() != access_type::invalid>>
auto transform(ExecutionPolicy p, buffer<T, Rank> first, buffer<V, Rank> second, buffer<U, Rank> out, const F &f)
{
	return transform(p, begin(first), end(first), begin(second), begin(out), f);
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
{
	return scoped_sequence{actions::generate(p, beg, end, f), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T, int Rank, typename F>
auto generate(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
{
	return generate(p, begin(in), end(in), f);
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const T &value)
{
	return scoped_sequence{actions::fill(p, beg, end, value), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename T, int Rank>
auto fill(ExecutionPolicy p, buffer<T, Rank> in, const T &value)
{
	return fill(p, begin(in), end(in), value);
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, T init, const BinaryOp &op)
{
	return scoped_sequence{actions::accumulate(p, beg, end, init, op), submit_to(p.q)};
}

template <typename ExecutionPolicy, typename BinaryOp, typename T, int Rank>
auto accumulate(ExecutionPolicy p, buffer<T, Rank> in, T init, const BinaryOp &op)
{
	return accumulate(p, begin(in), end(in), init, op);
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

} // namespace celerity::algorithm

#endif
