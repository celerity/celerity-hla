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

				static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

				const auto r = *end - *beg;
				assert(r <= static_cast<int>(out.buffer().size() - *out));

				return [=](celerity::handler cgh)
				{
					const auto in_acc = get_access< celerity::access_mode::read, InputAccessorType>(cgh, beg, end);
					auto out_acc = get_access<celerity::access_mode::write, OutputAccessorType>(cgh, beg, end);

					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(cl::sycl::range<Rank>{r}, [&](auto item)
							{
								out_acc[item] = f(in_acc[item]);
							});
					}
					else
					{
						cgh.run([&]()
							{
								std::for_each(beg, end,
									[&](auto i)
									{
										const cl::sycl::item<Rank> item{i};
										out_acc[item] = f(in_acc[item]);
									});
							});
					}
				};
			}

			template<typename ExecutionPolicy, typename F, typename T, size_t Rank>
			auto fill(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, const F& f)
			{
				using execution_policy = std::decay_t<ExecutionPolicy>;

				static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

				const auto r = *end - *beg;

				return [=](celerity::handler cgh)
				{
					auto out_acc = get_access<celerity::access_mode::write, celerity::algorithm::access_type::one_to_one>(cgh, beg, end);
	
					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(cl::sycl::range<Rank>{r}, [&](auto item)
							{
								out_acc[item] = f();
							});
					}
					else
					{
						cgh.run([&]()
							{
								std::for_each(beg, end,
									[&](auto i)
									{
										const cl::sycl::item<Rank> item{ i };
										out_acc[item] = f();
									});
							});
					}
				};
			}
		}

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F, 
			std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::slice, int> = 0>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f, size_t slice_dim)
		{
			return detail::transform<access_type::slice, access_type::one_to_one>(p, beg, end, out, f);
		}

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F,
			std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::one_to_one, int> = 0>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F & f)
		{
			return detail::transform<access_type::one_to_one, access_type::one_to_one>(p, beg, end, out, f);
		}

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F,
			std::enable_if_t<algorithm::detail::get_accessor_type<F, 0>() == access_type::chunk, int> = 0>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F & f, cl::sycl::range<Rank> chunk_size)
		{
			return detail::transform<access_type::chunk, access_type::one_to_one>(p, beg, end, out, f);
		}
	
		template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
		auto fill(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, const F & f)
		{
			return detail::fill(p, beg, end, f);
		}
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F, typename...Args>
	void transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f, Args...args)
	{
		task(actions::transform(p, beg, end, out, f, args...)) | submit_to(p.q);
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
	void fill(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, const F& f)
	{
		task(actions::fill(p, beg, end, f)) | submit_to(p.q);
	}
}

#endif
