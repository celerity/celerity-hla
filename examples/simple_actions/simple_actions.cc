#include "../../src/algorithm.h"
#include "../../src/actions.h"

// Use define instead of constexpr as MSVC seems to have some trouble getting it into nested closure
constexpr auto DEMO_DATA_SIZE = 10;

int main(int argc, char *argv[])
{
	// std::this_thread::sleep_for(std::chrono::seconds(5));
	auto verification_passed = true;

	using namespace celerity;

	try
	{
		distr_queue queue;

		buffer<float, 1> buf_a(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_b(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_c(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_d(cl::sycl::range<1>{DEMO_DATA_SIZE});

		using namespace algorithm;

		const auto verify = [&](float sum) {
			return [=, &verification_passed]() mutable {
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
		};

		{
			auto sum =
				fill(distr<class produce_a_1>(queue), buf_a, 1.f) |
				transform(distr<class compute_b_1>(queue), buf_a, buf_b, [](float x) { return 2.f * x; }) |
				transform(master(queue), buf_a, buf_c, [](const float x) { return 2.f - x; }) |
				transform(distr<class compute_d_1>(queue), buf_b, buf_c, buf_d, std::plus<float>{}) |
				accumulate(master_blocking(queue), buf_d, 0.0f, std::plus<float>{});

			actions::on_master(verify(sum));
		}

		//

		{
			auto produce_a = fill(distr<class produce_a_2>(queue), begin(buf_a), end(buf_a), 1.f);
			auto compute_b = transform(distr<class compute_b_2>(queue), begin(buf_a), end(buf_a), begin(buf_b), [](const float x) { return 2.f * x; });
			auto compute_c = transform(master(queue), begin(buf_a), end(buf_a), begin(buf_c), [](const float x) { return 2.f - x; });
			auto compute_d = transform(distr<class compute_d_2>(queue), begin(buf_b), end(buf_b), begin(buf_c), begin(buf_d), std::plus<float>{});
			auto reduce_d = accumulate(master_blocking(queue), begin(buf_d), end(buf_d), 0.0f, std::plus<float>{});

			actions::on_master(verify(produce_a | compute_b | compute_c | compute_d | reduce_d));
		}
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
