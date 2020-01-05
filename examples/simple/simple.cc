#include "../../src/algorithm.h"
#include "../../src/actions.h"

// Use define instead of constexpr as MSVC seems to have some trouble getting it into nested closure
constexpr auto DEMO_DATA_SIZE = 10;

int main(int argc, char *argv[])
{
	// std::this_thread::sleep_for(std::chrono::seconds(5));
	bool verification_passed = true;

	using namespace celerity;

	try
	{
		distr_queue queue;
		buffer<float, 1> buf_a(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_b(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_c(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_d(cl::sycl::range<1>{DEMO_DATA_SIZE});

		buf_a |
			algorithm::fill<class produce_a>(queue, {}, 1.f) |
			algorithm::transform<class compute_c>(queue, {}, buf_c, [](float x) { return 2.f - x; }) |
			algorithm::transform<class compute_b>(queue, buf_a, buf_b, [](float x) { return 2.f * x; }) | // parallel
			algorithm::transform<class compute_d>(queue, {}, {}, buf_d, std::plus<float>{}) << buf_c;

		transform(algorithm::master(queue), begin(buf_a), end(buf_a), begin(buf_c), [](float x) { return 2.f - x; });

		algorithm::master_task(algorithm::master(queue), [=, &verification_passed](auto &cgh) {
			auto r_d = buf_d.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(DEMO_DATA_SIZE));

			return [=, &verification_passed]() {
				size_t sum = 0;
				for (int i = 0; i < DEMO_DATA_SIZE; ++i)
				{
					sum += (size_t)r_d[i];
				}

				std::cout << "## RESULT: ";
				if (sum == 3 * DEMO_DATA_SIZE)
				{
					std::cout << "Success! Correct value was computed." << std::endl;
				}
				else
				{
					std::cout << "Fail! Value is " << sum << std::endl;
					verification_passed = false;
				}
			};
		});

		queue.slow_full_sync();
	}
	catch (std::exception &e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (cl::sycl::exception &e)
	{
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
