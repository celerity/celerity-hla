#ifndef POLICY_H
#define POLICY_H

#include "celerity_helper.h"

namespace celerity::algorithm
{

struct distributed_execution_policy
{
};

template <typename KernelName>
struct named_distributed_execution_policy : distributed_execution_policy
{
};

template <typename KernelName>
struct named_distributed_execution_and_queue_policy : named_distributed_execution_policy<KernelName>
{
	explicit named_distributed_execution_and_queue_policy(distr_queue &queue) : q(queue) {}
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

template <typename Policy>
struct policy_traits;

template <>
struct policy_traits<non_blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = false;
};

template <>
struct policy_traits<blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = true;
};

template <>
struct policy_traits<distributed_execution_policy>
{
	static constexpr bool is_distributed = true;
	static constexpr bool is_blocking = false;
};

template <typename KernelName>
struct policy_traits<named_distributed_execution_policy<KernelName>>
{
	static constexpr bool is_distributed = true;
	using kernel_name = KernelName;
};

template <typename T>
struct decay_policy
{
	using type = std::decay_t<T>;
};

template <typename KernelName>
struct decay_policy<named_distributed_execution_policy<KernelName>>
{
	using type = distributed_execution_policy;
};

template <typename KernelName>
struct decay_policy<named_distributed_execution_and_queue_policy<KernelName>>
{
	using type = named_distributed_execution_policy<KernelName>;
};

template <typename T>
using decay_policy_t = typename decay_policy<T>::type;

template <typename T>
struct strip_queue
{
	using type = T;
};

template <typename KernelName>
struct strip_queue<named_distributed_execution_and_queue_policy<KernelName>>
{
	using type = decay_policy_t<named_distributed_execution_and_queue_policy<KernelName>>;
};

template <typename T>
using strip_queue_t = typename strip_queue<T>::type;

template <typename KernelName>
auto distr(::celerity::distr_queue q) { return named_distributed_execution_and_queue_policy<KernelName>{q}; }

template <typename KernelName>
auto distr() { return named_distributed_execution_policy<KernelName>{}; }

inline auto master(celerity::distr_queue q) { return non_blocking_master_execution_policy{q}; }
inline auto master_blocking(celerity::distr_queue q) { return blocking_master_execution_policy{q}; }

} // namespace celerity::algorithm

#endif // POLICY_H
