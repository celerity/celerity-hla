#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wreturn-type"

#include "../../src/algorithm.h"
#include "../../src/actions.h"
#include "../../src/buffer_traits.h"

#include <numeric>

constexpr auto MAT_SIZE = 1024;

using namespace celerity;
using namespace algorithm;

namespace kernels
{
constexpr auto gen_a = [](cl::sycl::item<2> item) {
	return static_cast<float>(item.get_id(0) == item.get_id(1));
};

constexpr auto gen_b = [](cl::sycl::item<2> item) {
	return gen_a(item) * 2;
};

constexpr auto multiply = [](const slice_f<1> &a, const slice_f<0> &b) {
	return std::inner_product(begin(a), end(a), begin(b), 0.f);
};
} // namespace kernels

int main(int argc, char *argv[])
{
	// std::this_thread::sleep_for(std::chrono::seconds(5));
	celerity::runtime::init(&argc, &argv);
	bool verification_passed = true;

	celerity::experimental::bench::log_user_config({{"matSize", std::to_string(MAT_SIZE)}});

	using namespace celerity;
	using namespace algorithm;

	try
	{
		using buffer_type = celerity::buffer<float, 2>;
		using t = celerity::algorithm::buffer_traits<float, 2>;

		distr_queue queue;
		const auto mat_range = cl::sycl::range<2>{MAT_SIZE, MAT_SIZE};

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		auto mat_b =
			generate<class _1>(mat_range, kernels::gen_b) |
			submit_to(queue);

		auto out_buf =
			generate<class _2>(mat_range, kernels::gen_a) |
			transform<class _3>(kernels::multiply) << mat_b |
			transform<class _4>(kernels::multiply) << mat_b |
			submit_to(queue);

		master_task(algorithm::master(queue), [=, &verification_passed](auto &cgh) {
			auto r_d = out_buf.get_access<cl::sycl::access::mode::read>(cgh, out_buf.get_range());

			return [=, &verification_passed]() {
				celerity::experimental::bench::end("main program");

				for (size_t i = 0; i < MAT_SIZE; ++i)
				{
					for (size_t j = 0; j < MAT_SIZE; ++j)
					{
						const float correct_value = (i == j) * 4;

						if (r_d[{i, j}] == correct_value)
							continue;

						fprintf(stderr, "VERIFICATION FAILED for element %llu,%llu: %f != %f\n", i, j, r_d[{i, j}], correct_value);
						verification_passed = false;
					}
				}

				if (verification_passed)
				{
					printf("VERIFICATION PASSED!\n");
				}
			};
		});

		queue.slow_full_sync();
	}
	catch (cl::sycl::exception &e)
	{
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (std::exception &e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
