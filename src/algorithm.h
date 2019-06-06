#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "iterator.h"
#include "task.h"
#include "task_sequence.h"

namespace algorithm
{
	namespace tasks
	{
		template<typename T, size_t Dims, typename F>
		auto transform(iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
		{
			const auto r = *end - *beg;
			assert(r <= static_cast<int>(out.buffer().size() - *out));

			return [=](handler cgh)
			{
				auto in_acc = get_access(cgh,
					iterator_wrapper<T, Dims, iterator_type::one_to_one, access_mode::read>{ beg },
					iterator_wrapper<T, Dims, iterator_type::one_to_one, access_mode::read>{ end });

				auto out_acc = get_access(cgh,
					iterator_wrapper<T, Dims, iterator_type::one_to_one, access_mode::write>{ out },
					iterator_wrapper<T, Dims, iterator_type::one_to_one, access_mode::write>{ out });

				cgh.parallel_for<class transform>(range<1>{r}, [&](auto item)
					{
						out_acc[item] = f(in_acc[item]);
					});
			};
		}
	}

	struct celerity_distributed_execution_policy
	{
		distr_queue q;
	};

	struct celerity_master_execution_policy
	{
		distr_queue q;
	};

	auto dist(distr_queue q) { return celerity_distributed_execution_policy{ q }; }
	auto master(distr_queue q) { return celerity_master_execution_policy{ q }; }

	template<typename T, size_t Dims, typename F>
	void transform(celerity_distributed_execution_policy p, iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
	{
		::task(tasks::transform(beg, end, out, f)) | submit_to(p.q);
	}

	template<typename T, size_t Dims, typename F>
	void transform(celerity_master_execution_policy p, iterator<T, Dims> beg, iterator<T, Dims> end, iterator<T, Dims> out, const F& f)
	{
		// with master access
		::task(tasks::transform(beg, end, out, f)) | submit_to(p.q);
	}
}

#endif
