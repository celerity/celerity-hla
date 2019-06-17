#define MOCK_CELERITY

#include "../../src/sequence.h"
#include "../../src/actions.h"
#include "../../src/task_sequence.h"
#include "../../src/kernel_sequence.h"
#include "../../src/kernel.h"
#include "../../src/kernel_traits.h"
#include "../../src/static_iterator.h"
#include "../../src/algorithm.h"

#include <iostream>

using namespace std;

template<typename T, typename U>
void sequence_static_assertions(T zero, U hello_world)
{
	using namespace celerity;
	using namespace sequencing;

	using zero_t = T;
	using hello_world_t = U;

	static_assert(std::is_same<
		decltype(zero | task(zero)),
		sequence<task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | zero),
		kernel_sequence<zero_t, zero_t>>::value,
		"actions not promoted to kernel_sequence");

	static_assert(std::is_same<
		decltype(zero | zero | zero),
		kernel_sequence<zero_t, zero_t, zero_t>>::value,
		"action not appended to kernel_sequence");

	static_assert(std::is_same<
		decltype(hello_world | zero),
		sequence<hello_world_t, task_t<zero_t>>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | hello_world),
		sequence<task_t<zero_t>, hello_world_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(task(zero)),
		task_t<zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(fuse(zero | zero)),
		task_t<zero_t, zero_t>>::value,
		"actions to fused");

	static_assert(std::is_same<
		decltype(fuse(zero | zero | zero)),
		task_t<zero_t, zero_t, zero_t>>::value,
		"actions to fused");

	static_assert(std::is_same<
		decltype(zero | zero | task(zero)),
		sequence<task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action sequence not promoted to task_t sequence");

  	static_assert(std::is_same<
		decltype(zero | task(zero) | zero),
		sequence<task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

  	static_assert(std::is_same<
		decltype(hello_world | zero | task(zero) | zero),
		sequence<hello_world_t, task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");
}

void sequence_examples()
{
	// example 1: generic action sequence

	using namespace celerity;
	using namespace sequencing;
	using namespace actions;

	int i = 0;
	hello_world() | incr(i) | incr(i) | incr(i) | dispatch();
	cout << i << endl << endl;

	// example 2: celerity task sequence

	auto zero = [](handler cgh)
	{
		/*
		auto dw_buf = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
			cgh.parallel_for<class zero>(buf.get_range(), [=](cl::sycl::item<2> item) { dw_buf[item] = 0.f; });
		*/

		cout << cgh.invocations << ": zero" << endl;
	};

	auto step = [](handler cgh)
	{
		/*
			auto rw_up = up.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
			auto r_u = u.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));

			const auto size = up.get_range();
			cgh.parallel_for<KernelName>(size, [=](cl::sycl::item<2> item) {
				// NOTE: We have to do some casting due to some weird ComputeCpp behavior - possibly a device compiler bug.
				// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-94 (psalz)
				const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
				const size_t my = item[0] > 0 ? static_cast<size_t>(static_cast<int>(item[0]) - 1) : item[0];
				const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
				const size_t mx = item[1] > 0 ? static_cast<size_t>(static_cast<int>(item[1]) - 1) : item[1];

				// NOTE: We have to copy delta here, again to avoid some ComputeCpp weirdness.
				cl::sycl::float2 delta2 = delta;
				const float lap = (dt / delta2.y()) * (dt / delta2.y()) * ((r_u[{py, item[1]}] - r_u[item]) - (r_u[item] - r_u[{my, item[1]}]))
								  + (dt / delta2.x()) * (dt / delta2.x()) * ((r_u[{item[0], px}] - r_u[item]) - (r_u[item] - r_u[{item[0], mx}]));
				rw_up[item] = Config::a * 2 * r_u[item] - Config::b * rw_up[item] + Config::c * lap;
			});
		*/

		cout << cgh.invocations << ": step" << endl;
	};

	queue q{};

	hello_world() | zero | fuse(step | step | step) | step | submit_to(q);

	std::cout << endl;

	// 

	buffer<float, 1> b{ {5} };
	buffer<float, 1> b_out{ {5 } };

	{
		using namespace algorithm;
		using namespace tasks;

		auto add_one = tasks::transform(algorithm::dist<class add_one>(q), begin(b), end(b), begin(b_out), [](float x) { return x + 1; });

		zero | fuse(step | step | step) | step | add_one | submit_to(q);
	}


	algorithm::transform(algorithm::dist<class add_five>(q), begin(b), end(b), begin(b_out), [](float x) { return x + 5; });
	
	// ASSERTIONS

	sequence_static_assertions(zero, hello_world());
}

void iterator_static_assertions()
{
	using namespace celerity::algorithm::fixed;

	static_assert(static_index<1>::rank == 1, "static_index rank");
	static_assert(static_index<1, 2, 3>::rank == 3, "static_index rank");

	static_assert(static_index<1>::components[0] == 1, "static_index::components");
	static_assert(static_index<1, 2>::components[0] == 1 &&
		static_index<1, 2>::components[1] == 2, "static_index::components");

	static_assert(is_same<static_index<1>, static_index<1>>::value, "static_index equality");
	static_assert(is_same<static_index<1, 3>, static_index<1, 3>>::value, "static_index equality");
	static_assert(!is_same<static_index<1, 3>, static_index<1, 2>>::value, "static_index inequality");

	static_assert(static_iterator<float, 1, 2, 3>::rank == 3, "static_iterator::rank");
	static_assert(is_same<static_iterator<float, 1, 2, 3>::value_type, float>::value, "static_iterator value_type");
	static_assert(is_same<static_iterator<float, 1, 2, 3>::index_type, static_index<1, 2, 3>>::value, "static_iterator::value_type");
	static_assert(is_same<static_iterator<float, 1, 2, 3>, static_iterator<float, 1, 2, 3>>::value, "static_iterator equality");
	static_assert(!is_same<static_iterator<float, 1, 2, 3>, static_iterator<float, 3, 3, 3>>::value, "static_iterator inequality");


	static_assert(static_view<1, static_iterator<float, 0>, static_iterator<float, 0>>::id == 1, "static_view::id");
	static_assert(static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>::rank == 2, "static_view::rank");

	static_assert(is_same<static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>,
		static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>>::value,
		"static_view::rank");

	static_assert(!is_same<static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>,
		static_view<1, static_iterator<float, 0, 1>, static_iterator<float, 1, 1>>>::value,
		"static_view::rank");

	static_assert(is_same<decltype(begin(celerity::buffer<float, 1>{{1}})), static_iterator<float, 0 >> ::value, "begin");

	static_assert(is_same<decltype(end(celerity::buffer<float, 1>{{1}})), static_iterator<float, 0 >> ::value, "end");

	static_assert(is_same<decltype(make_view<1>(celerity::buffer<float, 1>{{1}})), static_view<1, static_iterator<float, 0>, static_iterator<float, 0>> > ::value, "make_view return type");

	using namespace celerity;
	using namespace algorithm;
	using namespace sequencing;
	using namespace tasks;

	const auto src_view = fixed::make_view<1>(celerity::buffer<float, 1>{ {1}});
	const auto dst_view = fixed::make_view<2>(celerity::buffer<float, 1>{ {1}});

	auto kernel = transform(src_view, dst_view, [](float x) { return 2 * x; });

	static_assert(is_invocable_v<decltype(kernel), handler>, "kernel invocable with handler");
	static_assert(is_same_v<decltype(task(kernel)), task_t<decltype(kernel)>>, "is task");
	static_assert(is_invocable_v<decltype(task(kernel)), queue&>, "task(kernel) invocable with queue");

	/*
	{
	  const auto view = make_view<1>(buffer<float, 1>{});
	  const auto first_kernel = make_kernel(view, [](handler){});
	  const auto second_kernel = make_kernel(view, [](handler){});

	  static_assert(is_combinable_v<decltype(first_kernel), decltype(second_kernel)>, "is_combinable_v");
	}

	{
	  const auto first_view = make_view<1>(buffer<float, 1>{});
	  const auto first_kernel = make_kernel(first_view, [](handler){});

	  const auto second_view = make_view<2>(buffer<float, 2>{});
	  const auto second_kernel = make_kernel(second_view, [](handler){});

	  static_assert(!is_combinable_v<decltype(first_kernel), decltype(second_kernel)>, "!is_combinable_v");
	}*/
}

int main() {

	iterator_static_assertions();
	sequence_examples();

	cout << endl;
	cin.get();

	return 0;
}