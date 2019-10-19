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

	template<typename F, std::enable_if_t<sizeof...(Actions) == 1, int> = 0>
	task_t(F f) : sequence_(std::move(f)) {}

	void operator()(distr_queue &q) const
	{
		using ret_type = std::invoke_result_t<decltype(sequence_), handler&>;

		q.submit([seq = sequence_](handler &cgh) 
		{
			if constexpr (contains_kernel_sequence_v<ret_type>)
			{
				auto kernels = std::invoke(seq, cgh);
				auto kernel_seq = kernel_sequence(sequence(kernels));

				std::invoke(kernel_seq, cgh);
			}
			else
			{
				std::invoke(seq, cgh);
			}
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

	decltype(auto) operator()(distr_queue &q) const
	{
		using ret_type = std::invoke_result_t<decltype(sequence_), handler &>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([seq = sequence_](handler &cgh) {
				std::invoke(seq, cgh);
			});
		}
		if constexpr (is_kernel_v<ret_type>)
		{
			using kernel_ret_type = std::invoke_result_t<ret_type, handler&>;

			if constexpr(std::is_void_v<ret_type>)
			{
				q.with_master_access([seq = sequence_](handler &cgh) {
					auto kernel = std::invoke(seq, cgh);
					std::invoke(kernel, cgh);
				});		
			}
			else
			{
				static_assert(std::is_void_v<kernel_ret_type>, "tasks may not return values  due to constness restrictions on master task");
			}		
		}
		else if constexpr (contains_kernel_sequence_v<ret_type>)
		{
			using kernel_ret_type = std::invoke_result_t<ret_type, handler&>;

			if constexpr(std::is_void_v<ret_type>)
			{
				q.with_master_access([seq = sequence_](handler &cgh) {
					auto kernels = std::invoke(seq, cgh);
					auto kernel_seq = kernel_sequence(sequence(kernels));

					std::invoke(kernel_seq, cgh);
				});
			}		
			else
			{
				static_assert(std::is_void_v<kernel_ret_type>, "tasks may not return values  due to constness restrictions on master task");
			}	
		}
		else
		{
			static_assert(std::is_void_v<ret_type>, "tasks may not return values  due to constness restrictions on master task");

			/*
			std::promise<ret_type> ret_value{};
			auto future = ret_value.get_future();

			q.with_master_access([&, promise = std::move(ret_value)](auto cgh) mutable {
				promise.set_value(std::invoke(sequence_, cgh));
			});

			return future;
			*/
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
			using kernel_ret_type = std::invoke_result_t<ret_type, handler&>;

			if constexpr(std::is_void_v<ret_type>)
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
			using kernel_ret_type = std::invoke_result_t<ret_type, handler&>;

			if constexpr(std::is_void_v<ret_type>)
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