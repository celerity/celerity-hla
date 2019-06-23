#ifndef TASK_H
#define TASK_H

#include "celerity.h"
#include "kernel_sequence.h"
#include "policy.h"

namespace celerity::algorithm
{

template<bool Distributed, typename...Actions>
class task_t;

template<typename...Actions>
class task_t<true, Actions...>
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
class task_t<true, F>
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
class task_t<false, F>
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
			ret_type ret_value{};

			q.with_master_access([&](auto cgh)
				{
					ret_value = std::invoke(sequence_, cgh);
				});

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
	return task_t<true, Actions...> { std::move(seq) };
}

template<typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T& invocable)
{
	return task_t<true, T>{ invocable };
}

template<bool Distributed, typename T>
auto task(const task_t<Distributed, T>& t)
{
	return t;
}

template<typename ExecutionPolicy, typename T, typename = std::enable_if_t<is_kernel_v<T>>>
auto task(const T& invocable)
{
	return task_t<policy_traits<std::decay_t<ExecutionPolicy>>::is_distributed, T>{invocable};
}

}

#endif