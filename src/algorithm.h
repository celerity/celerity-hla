#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "iterator.h"
#include "task.h"
#include "task_sequence.h"
#include "accessor_proxy.h"
#include "policy.h"

namespace celerity::algorithm
{
	namespace actions
	{
		namespace detail
		{
			template<access_type InputAccessorType, access_type OutputAccessorType, typename ExecutionPolicy, typename F, typename T,  size_t Rank>
			auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
			{
				using execution_policy = std::decay_t<ExecutionPolicy>;

				const auto r = *end - *beg;
				assert(r <= static_cast<int>(out.buffer().size() - *out));

				return [=](celerity::handler cgh)
				{
					const auto in_acc = get_access< celerity::access_mode::read, InputAccessorType>(cgh, beg, end);
					auto out_acc = get_access<celerity::access_mode::write, OutputAccessorType>(cgh, beg, end);

					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(celerity::range<Rank>{r}, [&](auto item)
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

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
		{
			constexpr auto input_accessor_type = algorithm::detail::get_accessor_type<Rank, F, 0>();
			constexpr auto output_accessor_type = access_type::one_to_one;

			return detail::transform<input_accessor_type, output_accessor_type>(p, beg, end, out, f);
		}
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
	void transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
	{
		task(actions::transform(p, beg, end, out, f)) | submit_to(p.q);
	}
}

#endif
