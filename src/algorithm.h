#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "iterator.h"
#include "task.h"
#include "task_sequence.h"
#include "accessor_proxy.h"
#include "policy.h"

namespace celerity::algorithm
{
	namespace tasks
	{
		template<typename T, size_t Dims, typename F, typename ExecutionPolicy>
		auto transform(ExecutionPolicy p, iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
		{
			using execution_policy = std::decay_t<ExecutionPolicy>;
			
			const auto r = *end - *beg;
			assert(r <= static_cast<int>(out.buffer().size() - *out));

			return [=](celerity::handler cgh)
			{
				const auto in_acc = get_access(cgh,
					iterator_wrapper<T, Dims, iterator_type::one_to_one, celerity::access_mode::read>{ beg },
					iterator_wrapper<T, Dims, iterator_type::one_to_one, celerity::access_mode::read>{ end });

				auto out_acc = get_access(cgh,
					iterator_wrapper<T, Dims, iterator_type::one_to_one, celerity::access_mode::write>{ out },
					iterator_wrapper<T, Dims, iterator_type::one_to_one, celerity::access_mode::write>{ out });

				if constexpr(policy_traits<execution_policy>::is_distributed)
				{
					cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(celerity::range<1>{r}, [&](auto item)
					{
						out_acc[item] = f(in_acc[item]);
					});
				}
				else
				{
					cgh.run(in_acc[item<1>{0}]);
				}
			};
		}
	}

	template<typename KernelName>
	auto dist(celerity::queue q) { return distributed_execution_policy<KernelName>{ q }; }

	inline auto master(celerity::queue q) { return celerity::master_execution_policy{ q }; }

	template<typename T, size_t Dims, typename F, typename KernelName>
	void transform(distributed_execution_policy<KernelName> p, iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
	{
		sequencing::task(tasks::transform(p, beg, end, out, f)) | sequencing::submit_to(p.q);
	}

	template<typename T, size_t Dims, typename F>
	void transform(master_execution_policy p, iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
	{
		// with master access
		sequencing::task(tasks::transform(p, beg, end, out, f)) | sequencing::submit_to(p.q);
	}
}

#endif
