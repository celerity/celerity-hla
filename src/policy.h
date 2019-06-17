#ifndef POLICY_H
#define POLICY_H

#include "celerity.h"

namespace celerity
{
	
template<class KernelName>
struct distributed_execution_policy
{
	queue q;
};

struct master_execution_policy
{
	queue q;
};

template<class Policy>
struct policy_traits;

template<>
struct policy_traits<master_execution_policy>
{
	static constexpr bool is_distributed = false;
	using kernel_name = void;
};

template<class KernelName>
struct policy_traits<distributed_execution_policy<KernelName>>
{
	static constexpr bool is_distributed = true;
	using kernel_name = KernelName;
};

}

#endif // POLICY_H

