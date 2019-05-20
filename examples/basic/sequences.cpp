#include "../../src/sequence.h"
#include <iomanip>
using namespace std;

int main() {

	hello_world() | hello_world() | []() { cout << "Hello Lambda" << endl; } | task([](/*cgh*/) {

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
	
	}) | submit_to(distr_queue{}) | dispatch();

	int i = 0;

	auto seq = incr(i) | incr(i);

	invoke(seq);

	cout << i << endl;

	cin.get();

	return i;
}