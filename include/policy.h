#ifndef POLICY_H
#define POLICY_H

#include "celerity_helper.h"
#include "policy_traits.h"

namespace celerity::algorithm
{

namespace detail
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

} // namespace detail

template <typename KernelName>
auto distr(::celerity::distr_queue q) { return detail::named_distributed_execution_and_queue_policy<KernelName>{q}; }

template <typename KernelName>
auto distr() { return detail::named_distributed_execution_policy<KernelName>{}; }

inline auto master(celerity::distr_queue q) { return detail::non_blocking_master_execution_policy{q}; }
inline auto master_blocking(celerity::distr_queue q) { return detail::blocking_master_execution_policy{q}; }

namespace traits
{

template <typename KernelName>
struct decay_policy<detail::named_distributed_execution_policy<KernelName>>
{
	using type = detail::distributed_execution_policy;
};

template <typename KernelName>
struct decay_policy<detail::named_distributed_execution_and_queue_policy<KernelName>>
{
	using type = detail::named_distributed_execution_policy<KernelName>;
};

template <typename KernelName>
struct strip_queue<detail::named_distributed_execution_and_queue_policy<KernelName>>
{
	using type = traits::decay_policy_t<detail::named_distributed_execution_and_queue_policy<KernelName>>;
};

template <>
struct policy_traits<detail::non_blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = false;
};

template <>
struct policy_traits<detail::blocking_master_execution_policy>
{
	static constexpr bool is_distributed = false;
	static constexpr bool is_blocking = true;
};

template <>
struct policy_traits<detail::distributed_execution_policy>
{
	static constexpr bool is_distributed = true;
	static constexpr bool is_blocking = false;
};

template <typename KernelName>
struct policy_traits<detail::named_distributed_execution_policy<KernelName>>
{
	static constexpr bool is_distributed = true;
	using kernel_name = KernelName;
};

} // namespace traits

} // namespace celerity::algorithm

template <typename T, size_t Id>
struct index_kernel_name_terminator
{
	using kernel_name = typename celerity::algorithm::traits::policy_traits<T>::kernel_name;
};

template <typename T, size_t Id = 0>
struct indexed_kernel_name
{
	using type = index_kernel_name_terminator<T, Id + 1>;
};

template <typename T, size_t Id>
struct indexed_kernel_name<indexed_kernel_name<T, Id>, Id>
{
	using type = indexed_kernel_name<T, Id + 1>;
};

namespace celerity::algorithm::traits
{

template <typename T, size_t Id>
struct policy_traits<index_kernel_name_terminator<T, Id>>
{
	using kernel_name = typename index_kernel_name_terminator<T, Id>::kernel_name;
};

} // namespace celerity::algorithm::traits

template <typename Policy>
using indexed_kernel_name_t = typename celerity::algorithm::traits::policy_traits<typename indexed_kernel_name<Policy>::type>::kernel_name;

template <typename FirstKernel, typename SecondKernel>
struct fused
{
};

namespace celerity::algorithm::traits
{
template <typename FirstKernel, typename SecondKernel>
struct policy_traits<fused<FirstKernel, SecondKernel>>
{
	using kernel_name = fused<FirstKernel, SecondKernel>;
};
} // namespace celerity::algorithm::traits

#endif // POLICY_H
