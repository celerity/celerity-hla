#ifndef KERNEL_SEQUENCE_H
#define KERNEL_SEQUENCE_H

#include "celerity.h"

template<typename...Actions>
class kernel_sequence
{
public:
	kernel_sequence(sequence<Actions...>&& s)
		: sequence_(std::move(s)) { }

	void operator()(distr_queue& q) const
	{
		std::cout << "queue.submit([](handler cgh){" << std::endl;
		q.submit([&](auto cgh) { std::invoke(sequence_, cgh); });
		std::cout << "});" << std::endl << std::endl;
	}

	auto sequence() { return sequence_; }

private:
	::sequence<Actions...> sequence_;
};

template<typename...Ts, typename...Us>
auto operator | (kernel_sequence<Ts...>&& lhs, kernel_sequence<Us...>&& rhs)
{
	auto seq = lhs.sequence() | rhs.sequence();
	return kernel_sequence<Ts..., Us...>{ seq };
}

template<typename...Actions>
struct sequence_traits<kernel_sequence<Actions...>>
{
	using is_sequence_type = std::integral_constant<bool, true>;
};

#endif