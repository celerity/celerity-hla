#ifndef TASK_H
#define TASK_H

#include "celerity_helper.h"
#include "sequence.h"
#include "policy.h"
#include "kernel_traits.h"
#include "iterator.h"
#include "accessor_type.h"
#include "require.h"

#include <future>

namespace celerity::hla
{

	namespace detail
	{

		template <typename ExecutionPolicy, typename... Actions>
		class task_t;

		template <typename KernelName, typename... Actions>
		class task_t<named_distributed_execution_policy<KernelName>, Actions...>
		{
		public:
			using execution_policy_type = named_distributed_execution_policy<KernelName>;

			static_assert(((traits::is_compute_task_v<Actions>)&&...), "task can only contain compute task functors");

			explicit task_t(sequence<Actions...> &&s)
				: sequence_(std::move(s)) {}

			explicit task_t(const sequence<Actions...> &s)
				: sequence_(s) {}

			task_t(Actions... f) : sequence_(std::move(f)...) {}

			template <int Rank>
			void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
			{
				using namespace traits;
				using namespace std;

				const auto d = distance(beg, end);

				q.submit(celerity::allow_by_ref, [seq = sequence_, d, beg](handler &cgh) {
					const auto r = invoke(seq, cgh);

					using first_kernel_type = first_result_t<decltype(r)>;
					using item_context_type = decay_t<arg_type_t<first_kernel_type, 0>>;

					static_assert(size_v<decltype(sequence(r))> == 1);

					cgh.template parallel_for<KernelName>(d, *beg, [r](cl::sycl::item<Rank> item) {
						item_context_type ctx{item};
						invoke(sequence(r), ctx);
					});
				});
			}

			auto get_sequence() const { return sequence_; }

		private:
			sequence<Actions...> sequence_;
		};

		template <typename F>
		class task_t<non_blocking_master_execution_policy, F>
		{
		public:
			static_assert(traits::is_master_task_v<F>, "task can only contain master task functors");

			explicit task_t(F f) : sequence_(std::move(f)) {}

			template <int Rank>
			void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
			{
				using namespace traits;
				using namespace std;

				const auto d = distance(beg, end);

				q.submit(celerity::allow_by_ref, [seq = sequence_, d, beg, end](handler &cgh) {
					const auto r = invoke(seq, cgh);

					using first_kernel_type = first_result_t<decltype(r)>;
					using item_context_type = decay_t<arg_type_t<first_kernel_type, 0>>;

					cgh.host_task(celerity::on_master_node, distance(beg, end), [=]() {
						for_each_index(beg, end, d, *beg, [r](cl::sycl::item<Rank> item) {
							item_context_type ctx{item};
							invoke(sequence(r), ctx);
						});
					});
				});
			}

			void operator()(distr_queue &q) const
			{
				q.submit(celerity::allow_by_ref, [seq = sequence_](handler &cgh) {
					cgh.host_task(celerity::on_master_node, std::invoke(seq, cgh));
				});
			}

			auto get_sequence() const { return sequence_; }

		private:
			sequence<F> sequence_;
		};

		template <typename F>
		class task_t<blocking_master_execution_policy, F>
		{
		public:
			static_assert(traits::is_master_task_v<F>, "task can only contain master task functors");

			explicit task_t(F f) : sequence_(std::move(f)) {}

			template <int Rank>
			void operator()(distr_queue &q, iterator<Rank> beg, iterator<Rank> end) const
			{
				using namespace traits;
				using namespace std;

				const auto d = distance(beg, end);

				q.submit(celerity::allow_by_ref, [seq = sequence_, d, beg, end](handler &cgh) {
					const auto r = invoke(seq, cgh);

					using first_kernel_type = first_result_t<decltype(r)>;
					using item_context_type = decay_t<arg_type_t<first_kernel_type, 0>>;

					cgh.host_task(celerity::on_master_node, [=]() {
						for_each_index(beg, end, d, *beg, [r](cl::sycl::item<Rank> item) {
							item_context_type ctx{item};
							invoke(sequence(r), ctx);
						});
					});
				});

				q.slow_full_sync();
			}

			void operator()(distr_queue &q) const
			{
				q.submit(celerity::allow_by_ref, [seq = sequence_](handler &cgh) {
					cgh.host_task(celerity::on_master_node, std::invoke(seq, cgh));
				});

				q.slow_full_sync();
			}

			auto get_sequence() const { return sequence_; }

		private:
			sequence<F> sequence_;
		};

		template <typename ExecutionPolicy, typename T>
		auto task(const task_t<ExecutionPolicy, T> &t)
		{
			return t;
		}

		template <typename ExecutionPolicy, typename T,
				  require<!traits::is_sequence_v<T>,
						  traits::is_master_task_v<T>> = yes>
		auto task(const T &invocable)
		{
			return task_t<traits::strip_queue_t<ExecutionPolicy>, T>{invocable};
		}

		template <typename ExecutionPolicy, typename... Ts>
		auto task(const sequence<Ts...> &seq)
		{
			using policy_type = traits::strip_queue_t<ExecutionPolicy>;
			return task_t<policy_type, Ts...>{seq};
		}

	} // namespace detail

	namespace traits
	{

		template <typename F>
		struct is_task : std::bool_constant<false>
		{
		};

		template <typename ExecutionPolicy, typename... Actions>
		struct is_task<detail::task_t<ExecutionPolicy, Actions...>> : std::bool_constant<true>
		{
		};

		template <typename F>
		inline constexpr bool is_task_v = is_task<F>::value;

	} // namespace traits

	namespace detail
	{

		template <typename F>
		decltype(auto) operator|(task_t<non_blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
		{
			return std::invoke(lhs, queue);
		}

		template <typename F>
		decltype(auto) operator|(task_t<blocking_master_execution_policy, F> lhs, celerity::distr_queue queue)
		{
			return std::invoke(lhs, queue);
		}

	} // namespace detail

} // namespace celerity::hla

#endif