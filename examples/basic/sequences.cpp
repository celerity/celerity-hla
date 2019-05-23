#include "../../src/sequence.h"
#include "../../src/actions.h"
#include "../../src/task_sequence.h"
#include "../../src/kernel_sequence.h"

#include <iostream>

using namespace std;

int main() {

	// example 1: generic action sequence

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

	distr_queue q{};

	hello_world() | zero | fuse(step | step | step) | step | submit_to(q);


	using zero_t = decltype(zero);
	using hello_world_t = decltype(hello_world());

	static_assert(std::is_same<
		decltype(zero | task(zero)),
		sequence<task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | zero),
		kernel_sequence<zero_t, zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | zero | zero),
		kernel_sequence<zero_t, zero_t, zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(hello_world() | zero),
		sequence<hello_world_t, task_t<zero_t>>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | hello_world()),
		sequence<task_t<zero_t>, hello_world_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(task(zero)),
		task_t<zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(fuse(zero | zero)),
		task_t<zero_t, zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(fuse(zero | zero | zero)),
		task_t<zero_t, zero_t, zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | zero | task(zero)),
		sequence<task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

  static_assert(std::is_same<
		decltype(zero | task(zero) | zero),
		sequence<task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

  static_assert(std::is_same<
		decltype(hello_world() | zero | task(zero) | zero),
		sequence<hello_world_t, task_t<zero_t>, task_t<zero_t>, task_t<zero_t>>>::value,
		"action not promoted to task_t");

	cout << endl;

	cin.get();

	return i;
}