#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "iterator.h"
#include "task.h"
#include "task_sequence.h"
#include "accessor_proxy.h"
#include "policy.h"
#include <future>

namespace celerity::algorithm
{
	namespace actions
	{
		namespace detail
		{
			template<typename InputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T,  size_t Rank>
			auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
			{
				using execution_policy = std::decay_t<ExecutionPolicy>;

				static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

				const auto r = distance(beg, end);

				//assert(r <= static_cast<int>(out.buffer().size() - *out));

				return [=](celerity::handler cgh)
				{
					auto in_acc = get_access<celerity::access_mode::read, InputAccessorType>(cgh, beg, end);
					auto out_acc = get_access<celerity::access_mode::write, OutputAccessorType>(cgh, out, out);

					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(r, [&](auto item)
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

			template<typename FirstInputAccessorType, typename SecondInputAccessorType, typename OutputAccessorType, typename ExecutionPolicy, typename F, typename T, size_t Rank>
			auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> beg2, iterator<T, Rank> out, const F& f)
			{
				using execution_policy = std::decay_t<ExecutionPolicy>;

				const auto r = distance(beg, end);

				//assert(r <= static_cast<int>(beg2.buffer().size() - *beg2));
				//assert(r <= static_cast<int>(out.buffer().size() - *out));

				return [=](celerity::handler cgh)
				{
					auto first_in_acc = get_access<celerity::access_mode::read, FirstInputAccessorType>(cgh, beg, end);
					auto second_in_acc = get_access<celerity::access_mode::read, SecondInputAccessorType>(cgh, beg2, beg2);

					auto out_acc = get_access<celerity::access_mode::write, OutputAccessorType>(cgh, out, out);

					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(r, [&](auto item)
							{
								out_acc[item] = f(first_in_acc[item], second_in_acc[item]);
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
										out_acc[item] = f(first_in_acc[item], second_in_acc[item]);
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

				const auto r = distance(beg, end);

				return [=](celerity::handler cgh)
				{
					auto out_acc = get_access<celerity::access_mode::write, one_to_one>(cgh, beg, end);
	
					if constexpr (policy_traits<execution_policy>::is_distributed)
					{
						cgh.parallel_for<policy_traits<execution_policy>::kernel_name>(r, [&](auto item)
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
		
			template<typename ExecutionPolicy, typename BinaryOp, typename T, size_t Rank,
				typename = ::std::enable_if_t<algorithm::detail::get_accessor_type<BinaryOp, 1>() == access_type::one_to_one>>
			auto accumulate(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, T init, const BinaryOp& op)
			{
				static_assert(!policy_traits<ExecutionPolicy>::is_distributed, "can not be distributed");
				static_assert(Rank == 1, "Only 1-dimenionsal buffers for now");

				const auto r = distance(beg, end);

				return [=](celerity::handler cgh)
				{
					const auto in_acc = get_access<access_mode::read, one_to_one>(cgh, beg, end);

					auto sum = init;

					cgh.run([&]()
					{
						std::for_each(beg, end,
							[&](auto i)
							{
								const cl::sycl::item<Rank> item{ i };
								sum = op(std::move(sum), in_acc[item]);
							});
					});

					return sum;
				};
			}
		}

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
		{
			return task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>, one_to_one>(p, beg, end, out, f));
		}

		template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
		auto transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> beg2, iterator<T, Rank> out, const F& f)
		{
			return task<ExecutionPolicy>(detail::transform<algorithm::detail::accessor_type_t<F, 0, T>, algorithm::detail::accessor_type_t<F, 1, T>, one_to_one>(p, beg, end, beg2, out, f));
		}
	
		template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
		auto fill(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, const F & f)
		{
			return task<ExecutionPolicy>(detail::fill(p, beg, end, f));
		}
	
		template<typename ExecutionPolicy, typename BinaryOp, typename T, size_t Rank>
		auto accumulate(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, T init, const BinaryOp & op)
		{
			return task<ExecutionPolicy>(detail::accumulate(p, beg, end, init, op));
		}
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F,
		typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid>>
	void transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> out, const F& f)
	{
		actions::transform(p, beg, end, out, f) | submit_to(p.q);
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F,
		typename = std::enable_if_t<detail::get_accessor_type<F, 0>() != access_type::invalid &&
									detail::get_accessor_type<F, 1>() != access_type::invalid>>
	void transform(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, iterator<T, Rank> beg2, iterator<T, Rank> out, const F& f)
	{
		actions::transform(p, beg, end, beg2, out, f) | submit_to(p.q);
	}

	template<typename ExecutionPolicy, typename T, size_t Rank, typename F>
	void fill(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, const F& f)
	{
		actions::fill(p, beg, end, f) | submit_to(p.q);
	}

	template<typename ExecutionPolicy, typename BinaryOp, typename T, size_t Rank>
	auto accumulate(ExecutionPolicy p, iterator<T, Rank> beg, iterator<T, Rank> end, T init, const BinaryOp& op)
	{
		return actions::accumulate(p, beg, end, init, op) | submit_to(p.q);
	}
}

#endif
