#ifndef TASK_H
#define TASK_H

#include "celerity.h"
#include "kernel_sequence.h"

template<typename...Actions>
class task_t
{
public:
	explicit task_t(::kernel_sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }

	void operator()(distr_queue& q) const
	{
		sequence_(q);
	}

private:
	::kernel_sequence<Actions...> sequence_;
};

template<typename...Actions>
auto task(kernel_sequence<Actions...>&& seq)
{
	return task_t<Actions...> { std::move(seq) };
}

template<typename T>
auto task(const T& invocable)
{
	return task_t<T>{ {invocable}};
}


#endif