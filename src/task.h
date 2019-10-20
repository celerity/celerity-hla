#ifndef TASK_H
#define TASK_H

#include "celerity_helper.h"
#include "kernel_sequence.h"
#include "policy.h"
#include "kernel_traits.h"
#include "iterator.h"

#include <future>

namespace celerity::algorithm
{

template <typename T, std::enable_if_t<!detail::_is_kernel_v<T>, int> = 0>
auto to_kernel(T t)
{
	return sequence(t);
}

template <typename T, std::enable_if_t<detail::_is_kernel_v<T>, int> = 0>
auto to_kernel(T t)
{
	return t;
}

template <typename ExecutionPolicy, typename... Actions>
class task_t;

template <typename KernelName, typename... Actions>
class task_t<named_distributed_execution_policy<KernelName>, Actions...>
{
public:
	static_assert(((detail::_is_task_v<Actions>)&&...), "task can only contain task functors");

	explicit task_t(kernel_sequence<Actions...> &&s)
		: sequence_(std::move(s)) {}

	template <typename F, std::enable_if_t<sizeof...(Actions) == 1, int> = 0>
	task_t(F f) : sequence_(std::move(f)) {}

	template <int Rank>
	void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
	{
		const auto d = distance(beg, end);

		q.submit([seq = sequence_, d, beg](handler &cgh) {
			const auto r = std::invoke(seq, cgh);
			cgh.template parallel_for<KernelName>(d, *beg, to_kernel(r));
		});
	}

private:
	kernel_sequence<Actions...> sequence_;
};

template <typename F>
class task_t<non_blocking_master_execution_policy, F>
{
public:
	explicit task_t(F f) : sequence_(std::move(f)) {}

	template <int Rank>
	void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
	{
		const auto d = distance(beg, end);

		q.submit([seq = sequence_, d, beg, end](handler &cgh) {
			const auto r = std::invoke(seq, cgh);
			cgh.run([&]() { for_each_index(beg, end, d, *beg, to_kernel(r)); });
		});
	}

private:
	kernel_sequence<F> sequence_;
};

template <typename F>
class task_t<blocking_master_execution_policy, F>
{
public:
	explicit task_t(F f) : sequence_(std::move(f)) {}

	decltype(auto) operator()(distr_queue &q) const
	{
		using ret_type = std::invoke_result_t<decltype(sequence_), handler &>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([&](auto &cgh) {
				std::invoke(sequence_, cgh);
			});

			q.slow_full_sync();
		}
		else if constexpr (is_kernel_v<ret_type>)
		{
			using kernel_ret_type = std::invoke_result_t<ret_type, handler &>;

			if constexpr (std::is_void_v<ret_type>)
			{
				q.with_master_access([seq = sequence_](handler &cgh) {
					auto kernel = std::invoke(seq, cgh);
					std::invoke(kernel, cgh);
				});

				q.slow_full_sync();
			}
			else
			{
				kernel_ret_type ret_value{};

				q.with_master_access([&ret_value, seq = sequence_](handler &cgh) {
					auto kernel = std::invoke(seq, cgh);
					ret_value = std::invoke(kernel, cgh);
				});

				q.slow_full_sync();

				return ret_value;
			}
		}
		else if constexpr (contains_kernel_sequence_v<ret_type>)
		{
			using kernel_ret_type = std::invoke_result_t<ret_type, handler &>;

			if constexpr (std::is_void_v<ret_type>)
			{
				q.with_master_access([seq = sequence_](handler &cgh) {
					auto kernels = std::invoke(seq, cgh);
					auto kernel_seq = kernel_sequence(sequence(kernels));
					std::invoke(kernel_seq, cgh);
				});

				q.slow_full_sync();
			}
			else
			{
				kernel_ret_type ret_value{};

				q.with_master_access([&ret_value, seq = sequence_](handler &cgh) {
					auto kernels = std::invoke(seq, cgh);
					auto kernel_seq = kernel_sequence(sequence(kernels));
					ret_value = std::invoke(kernel_seq, cgh);
				});

				q.slow_full_sync();

				return ret_value;
			}
		}
		else
		{
			ret_type ret_value{};

			q.with_master_access([&](auto &cgh) {
				ret_value = std::invoke(sequence_, cgh);
			});

			q.slow_full_sync();

			return ret_value;
		}
	}

private:
	kernel_sequence<F> sequence_;
};

template <typename... Actions>
auto fuse(kernel_sequence<Actions...> &&seq)
{
	return task_t<distributed_execution_policy, Actions...>{std::move(seq)};
}

template <typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T &invocable)
{
	return task_t<distributed_execution_policy, T>{invocable};
}

template <typename ExecutionPolicy, typename T>
auto task(const task_t<ExecutionPolicy, T> &t)
{
	return t;
}

template <typename ExecutionPolicy, typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T &invocable)
{
	return task_t<decay_policy_t<ExecutionPolicy>, T>{invocable};
}

template <typename F>
struct is_task : std::bool_constant<false>
{
};

template <typename ExecutionPolicy, typename... Actions>
struct is_task<task_t<ExecutionPolicy, Actions...>> : std::bool_constant<true>
{
};

template <typename F>
inline constexpr bool is_task_v = is_task<F>::value;

} // namespace celerity::algorithm

#endif