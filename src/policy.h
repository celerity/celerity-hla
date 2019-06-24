#ifndef POLICY_H
#define POLICY_H

#include "celerity.h"

namespace celerity::algorithm
{

struct distributed_execution_policy
{
	
};
	
template<class KernelName>
struct named_distributed_execution_policy : distributed_execution_policy
{
	explicit named_distributed_execution_policy(distr_queue& queue) : q(queue) {}

	::celerity::distr_queue q;
};

struct non_blocking_master_execution_policy
{
	::celerity::distr_queue q;
};

struct blocking_master_execution_policy
{
	::celerity::distr_queue q;
};

template<class Policy>
struct policy_traits;

template<>
struct policy_traits<non_blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = false;
};

template<>
struct policy_traits<blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = true;
};

template<>
struct policy_traits<distributed_execution_policy>
{
	static constexpr bool is_distributed = true;
	static constexpr bool is_blocking = false;
};

template<class KernelName>
struct policy_traits<named_distributed_execution_policy<KernelName>>
{
	static constexpr bool is_distributed = true;
	using kernel_name = KernelName;
};

template<typename KernelName>
auto distr(::celerity::distr_queue q) { return named_distributed_execution_policy<KernelName>{q}; }

inline auto master(celerity::distr_queue q) { return non_blocking_master_execution_policy{ q }; }
inline auto master_blocking(celerity::distr_queue q) { return blocking_master_execution_policy{ q }; }

}

#endif // POLICY_H

