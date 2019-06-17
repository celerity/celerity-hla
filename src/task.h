#ifndef TASK_H
#define TASK_H

#include "celerity.h"
#include "kernel_sequence.h"

namespace celerity::sequencing
{

template<typename...Actions>
class task_t
{
public:
	explicit task_t(kernel_sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }

	void operator()(queue& q) const
	{
		std::invoke(sequence_, q);
	}

private:
	kernel_sequence<Actions...> sequence_;
};

template<typename F>
class task_t<F>
{
public:
	task_t(F f) : sequence_(std::move(f)) { }

	void operator()(celerity::queue& q) const
	{
		std::invoke(sequence_, q);
	}

private:
	kernel_sequence<F> sequence_;
};

template<typename...Actions>
auto fuse(kernel_sequence<Actions...>&& seq)
{
	return task_t<Actions...> { std::move(seq) };
}

template<typename T, typename = std::enable_if_t<traits::is_kernel_v<T>>>
auto task(const T& invocable)
{
	return task_t<T>{ invocable };
}

template<typename T>
auto task(const task_t<T>& t)
{
	return t;
}

}

#endif