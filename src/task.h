#ifndef TASK_H
#define TASK_H

#include "celerity_helper.h"
#include "sequence.h"
#include "policy.h"
#include "kernel_traits.h"
#include "iterator.h"
#include "accessor_type.h"

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
	using execution_policy_type = named_distributed_execution_policy<KernelName>;

	static_assert(((detail::is_compute_task_v<Actions>)&&...), "task can only contain compute task functors");

	explicit task_t(algorithm::sequence<Actions...> &&s)
		: sequence_(std::move(s)) {}

	explicit task_t(const algorithm::sequence<Actions...> &s)
		: sequence_(s) {}

	task_t(Actions... f) : sequence_(std::move(f)...) {}

	template <int Rank>
	void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
	{
		const auto d = distance(beg, end);

		q.submit([seq = sequence_, d, beg](handler &cgh) {
			const auto r = std::invoke(seq, cgh);
			cgh.template parallel_for<KernelName>(d, *beg, [=](cl::sycl::item<1> item){
				
				item_context<1, int> ctx{item};
				std::invoke(to_kernel(r), ctx); 
			});
		});
	}

	auto get_sequence() const { return sequence_; }

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

	auto get_sequence() const { return sequence_; }

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

	auto get_sequence() const { return sequence_; }

private:
	algorithm::sequence<F> sequence_;
};

template <typename ExecutionPolicy, typename T>
auto task(const task_t<ExecutionPolicy, T> &t)
{
	return t;
}

template <typename ExecutionPolicy, typename T, std::enable_if_t<!is_sequence_v<T> && detail::is_master_task_v<T>, int> = 0>
auto task(const T &invocable)
{
	return task_t<strip_queue_t<ExecutionPolicy>, T>{invocable};
}

template <typename ExecutionPolicy, typename... Ts>
auto task(const sequence<Ts...> &seq)
{
	using policy_type = strip_queue_t<ExecutionPolicy>;
	return task_t<policy_type, Ts...>{seq};
}

template <typename ExecutionPolicyA, typename KernelA, typename ExecutionPolicyB, typename KernelB>
auto fuse(task_t<ExecutionPolicyA, KernelA> a, task_t<ExecutionPolicyB, KernelB> b)
{
	using new_execution_policy = named_distributed_execution_policy<
	 	indexed_kernel_name_t<fused<ExecutionPolicyA, ExecutionPolicyB>>>;

	using kernel_type = std::invoke_result_t<decltype(a.get_sequence()), handler&>;
	using item_type = detail::arg_type_t<kernel_type, 0>;

    auto seq = a.get_sequence() | b.get_sequence();

    auto f = [=](handler& cgh)
    {
        auto kernels = sequence(std::invoke(seq, cgh));

        return [=](item_type item)
        {
            kernels(item);
        };
    };

	return task<new_execution_policy>(f);
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