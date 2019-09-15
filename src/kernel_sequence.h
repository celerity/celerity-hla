#ifndef KERNEL_SEQUENCE_H
#define KERNEL_SEQUENCE_H

#include "celerity.h"
#include "sequence.h"

namespace celerity::algorithm
{
template <typename... Actions>
class kernel_sequence
{
public:
	kernel_sequence(sequence<Actions...> &&s)
		: sequence_(std::move(s)) {}

	decltype(auto) operator()(handler cgh) const
	{
		return std::invoke(sequence_, cgh);
	}

	auto get_sequence() { return sequence_; }

private:
	celerity::algorithm::sequence<Actions...> sequence_;
};

template <typename... Ts, typename... Us>
auto operator|(kernel_sequence<Ts...> &&lhs, kernel_sequence<Us...> &&rhs)
{
	auto seq = lhs.sequence() | rhs.sequence();
	return kernel_sequence<Ts..., Us...>{seq};
}

template <typename... Actions>
struct sequence_traits<algorithm::kernel_sequence<Actions...>>
{
	using is_sequence_type = std::integral_constant<bool, true>;
};
} // namespace celerity::algorithm

#endif