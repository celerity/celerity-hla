#ifndef TASK_H
#define TASK_H

#include "celerity.h"
#include "kernel_sequence.h"
#include "policy.h"

#include <future>

namespace celerity::algorithm
{

template<typename ExecutionPolicy, typename...Actions>
class task_t;

template<typename...Actions>
class task_t<distributed_execution_policy, Actions...>
{
public:
	explicit task_t(kernel_sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }

	void operator()(distr_queue& q) const
	{
		std::cout << "queue.submit([](handler cgh){" << std::endl;
		q.submit([&](auto cgh) { std::invoke(sequence_, cgh); });
		std::cout << "});" << std::endl << std::endl;
	}

private:
	kernel_sequence<Actions...> sequence_;
};

template<typename F>
class task_t<distributed_execution_policy, F>
{
public:
	task_t(F f) : sequence_(std::move(f)) { }

	decltype(auto) operator()(distr_queue& q) const
	{
		std::cout << "queue.submit([](handler cgh){" << std::endl;
		q.submit([&](auto cgh) { std::invoke(sequence_, cgh); });
		std::cout << "});" << std::endl << std::endl;
	}

private:
	kernel_sequence<F> sequence_;
};

template<typename F>
class task_t<non_blocking_master_execution_policy, F>
{
public:
	explicit task_t(F f) : sequence_(std::move(f)) { }

	decltype(auto) operator()(distr_queue& q) const
	{
		std::cout << "queue.with_master_access([](handler cgh){" << std::endl;

		using ret_type = std::invoke_result_t<decltype(sequence_), handler>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([&](auto cgh)
				{
					std::invoke(sequence_, cgh);
				});

			std::cout << "});" << std::endl << std::endl;
		}
		else
		{
			std::promise<ret_type> ret_value{};
			auto future = ret_value.get_future();

			q.with_master_access([&, promise = std::move(ret_value)](auto cgh) mutable
				{
					promise.set_value(std::invoke(sequence_, cgh));
				});

			std::cout << "});" << std::endl << std::endl;

			return future;
		}
	}

private:
	kernel_sequence<F> sequence_;
};

template<typename F>
class task_t<blocking_master_execution_policy, F>
{
public:
	explicit task_t(F f) : sequence_(std::move(f)) { }

	decltype(auto) operator()(distr_queue& q) const
	{
		std::cout << "queue.with_master_access([](handler cgh){" << std::endl;

		using ret_type = std::invoke_result_t<decltype(sequence_), handler>;

		if constexpr (std::is_void_v<ret_type>)
		{
			q.with_master_access([&](auto cgh)
				{
					std::invoke(sequence_, cgh);
				});

			q.wait();

			std::cout << "});" << std::endl << std::endl;
		}
		else
		{
			ret_type ret_value{};

			q.with_master_access([&](auto cgh) mutable
			{
				ret_value = std::invoke(sequence_, cgh);
			});

			q.wait();

			std::cout << "});" << std::endl << std::endl;

			return ret_value;
		}
	}

private:
	kernel_sequence<F> sequence_;
};

template<typename...Actions>
auto fuse(kernel_sequence<Actions...>&& seq)
{
	return task_t<distributed_execution_policy, Actions...> { std::move(seq) };
}

template<typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T& invocable)
{
	return task_t<distributed_execution_policy, T>{ invocable };
}

template<typename ExecutionPolicy, typename T>
auto task(const task_t<ExecutionPolicy, T>& t)
{
	return t;
}

template<typename ExecutionPolicy, typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T& invocable)
{
	return task_t<ExecutionPolicy, T>{invocable};
}

}

#endif