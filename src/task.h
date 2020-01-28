#ifndef TASK_H
#define TASK_H

#include "celerity_helper.h"
#include "sequence.h"
#include "policy.h"
#include "kernel_traits.h"
#include "iterator.h"

#include <future>

namespace celerity::algorithm
{

template <typename T, std::enable_if_t<!detail::is_kernel_v<T>, int> = 0>
auto to_kernel(T t)
{
	return sequence(t);
}

template <typename T, std::enable_if_t<detail::is_kernel_v<T>, int> = 0>
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
	static_assert(((detail::is_compute_task_v<Actions>)&&...), "task can only contain compute task functors");

	explicit task_t(algorithm::sequence<Actions...> &&s)
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
	algorithm::sequence<Actions...> sequence_;
};

template <typename F>
class task_t<non_blocking_master_execution_policy, F>
{
public:
	static_assert(detail::is_master_task_v<F>, "task can only contain master task functors");

	explicit task_t(F f) : sequence_(std::move(f)) {}

	template <int Rank>
	void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
	{
		const auto d = distance(beg, end);

		q.with_master_access([seq = sequence_, d, beg, end](handler &cgh) {
			const auto r = std::invoke(seq, cgh);
			cgh.run([&]() { for_each_index(beg, end, d, *beg, to_kernel(r)); });
		});
	}

	void operator()(distr_queue &q) const
	{
		q.with_master_access([seq = sequence_](handler &cgh) {
			cgh.run(std::invoke(seq, cgh));
		});
	}

private:
	algorithm::sequence<F> sequence_;
};

template <typename F>
class task_t<blocking_master_execution_policy, F>
{
public:
	static_assert(detail::is_master_task_v<F>, "task can only contain master task functors");

	explicit task_t(F f) : sequence_(std::move(f)) {}

	template <int Rank>
	void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
	{
		const auto d = distance(beg, end);

		q.with_master_access([seq = sequence_, d, beg, end](handler &cgh) {
			const auto r = std::invoke(seq, cgh);
			cgh.run([&]() { for_each_index(beg, end, d, *beg, to_kernel(r)); });
		});

		q.slow_full_sync();
	}

	void operator()(distr_queue &q) const
	{
		q.with_master_access([seq = sequence_](handler &cgh) {
			cgh.run(std::invoke(seq, cgh));
		});

		q.slow_full_sync();
	}

private:
	algorithm::sequence<F> sequence_;
};

template <typename ExecutionPolicy, typename T>
auto task(const task_t<ExecutionPolicy, T> &t)
{
	return t;
}

template <typename ExecutionPolicy, typename T, typename = std::enable_if_t<detail::is_master_task_v<T>>>
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