#ifndef TASK_H
#define TASK_H

#include "celerity_helper.h"
#include "kernel_sequence.h"
#include "policy.h"

#include <future>

namespace celerity::algorithm
{

template <typename ExecutionPolicy, typename... Actions>
class task_t;

template <typename... Actions>
class task_t<distributed_execution_policy, Actions...>
{
public:
	explicit task_t(kernel_sequence<Actions...> &&s)
		: sequence_(std::move(s)) {}

	void operator()(distr_queue &q) const
	{
#ifdef DEBUG_
		std::cout << "queue.submit([](handler cgh){" << std::endl;
#endif

		auto f = [seq = sequence_](handler &cgh) { std::invoke(seq, cgh); };

		q.submit(f);

#ifdef DEBUG_
		std::cout << "});" << std::endl
				  << std::endl;
#endif
	}

private:
	kernel_sequence<Actions...> sequence_;
};

template <typename F>
class task_t<distributed_execution_policy, F>
{
public:
	task_t(F f) : sequence_(std::move(f)) {}

	void operator()(distr_queue &q) const
	{
#ifdef DEBUG_
		std::cout << "queue.submit([](handler cgh){" << std::endl;
#endif

		//q.submit([&](handler &cgh) { std::invoke(sequence_, cgh); });

		const auto f = [seq = sequence_](handler &cgh) { std::invoke(seq, cgh); };

		q.submit(f);

#ifdef DEBUG_
		std::cout << "});" << std::endl
				  << std::endl;
#endif
	}

private:
	kernel_sequence<F> sequence_;
};

template <typename F>
class task_t<non_blocking_master_execution_policy, F>
{
public:
	explicit task_t(F f) : sequence_(std::move(f)) {}

	decltype(auto) operator()(distr_queue &q) const
	{
#ifdef DEBUG_
		std::cout << "queue.with_master_access([](handler cgh){" << std::endl;
#endif

		using ret_type = std::invoke_result_t<decltype(sequence_), handler &>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([seq = sequence_](handler &cgh) {
				std::invoke(seq, cgh);
			});

#ifdef DEBUG_
			std::cout << "});" << std::endl
					  << std::endl;
#endif
		}
		else
		{
			std::promise<ret_type> ret_value{};
			auto future = ret_value.get_future();

			/*q.with_master_access([&, promise = std::move(ret_value)](auto cgh) mutable {
				promise.set_value(std::invoke(sequence_, cgh));
			});*/

#ifdef DEBUG_
			std::cout << "});" << std::endl
					  << std::endl;
#endif

			return future;
		}
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
#ifdef DEBUG_
		std::cout << "queue.with_master_access([](handler cgh){" << std::endl;
#endif

		using ret_type = std::invoke_result_t<decltype(sequence_), handler &>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([&](auto &cgh) {
				std::invoke(sequence_, cgh);
			});

			q.slow_full_sync();

#ifdef DEBUG_
			std::cout << "});" << std::endl
					  << std::endl;
#endif
		}
		else
		{
			ret_type ret_value{};

			q.with_master_access([&](auto &cgh) {
				ret_value = std::invoke(sequence_, cgh);
			});

			q.slow_full_sync();

#ifdef DEBUG_
			std::cout << "});" << std::endl
					  << std::endl;
#endif

			return ret_value;
		}
	}

private:
	kernel_sequence<F> sequence_;
};

template <typename KernelName, typename F>
class task_t<named_distributed_execution_policy<KernelName>, F> : public task_t<distributed_execution_policy, F>
{
	using base_type = task_t<distributed_execution_policy, F>;
	using base_type::base_type;
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