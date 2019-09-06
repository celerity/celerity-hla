#define MOCK_CELERITY
#include "../../src/algorithm.h"
#include "../../src/actions.h"

// Use define instead of constexpr as MSVC seems to have some trouble getting it into nested closure
constexpr auto DEMO_DATA_SIZE = 10;

int main(int argc, char* argv[]) {
	// std::this_thread::sleep_for(std::chrono::seconds(5));
	auto verification_passed = true;

	using namespace celerity;

	try {
		distr_queue queue;

		buffer<float, 1> buf_a(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_b(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_c(cl::sycl::range<1>{DEMO_DATA_SIZE});
		buffer<float, 1> buf_d(cl::sycl::range<1>{DEMO_DATA_SIZE});

		using namespace algorithm;
		using namespace actions;

		const auto verify = [&](auto future)
		{
			return[&, f = std::move(future)]() mutable
			{
				const auto sum = f.get();

				std::cout << "## RESULT: ";
				if (sum == 3 * DEMO_DATA_SIZE) {
					std::cout << "Success! Correct value was computed." << std::endl;
				}
				else {
					std::cout << "Fail! Value is " << sum << std::endl;
					verification_passed = false;
				}
			};
		};

		{
			auto sum_future =
				actions::fill(distr<class produce_a>(queue), begin(buf_a), end(buf_a), 1.f) |
				actions::transform(distr<class compute_b>(queue), begin(buf_a), end(buf_a), begin(buf_b), [](float x) { return 2.f * x; }) |
				actions::transform(master(queue), begin(buf_a), end(buf_a), begin(buf_c), [](const float x) { return 2.f - x; }) |
				actions::transform(distr<class compute_d>(queue), begin(buf_b), end(buf_b), begin(buf_c), begin(buf_d), [](const float x, const float y) { return x + y; }) |
				actions::accumulate(master(queue), begin(buf_d), end(buf_d), 0.0f, [](const float acc, const float x) { return acc + x; }) |
				submit_to(queue);

			on_master(verify(std::move(sum_future)));
		}

		// 

		{
			auto produce_a = actions::fill(distr<class produce_a>(queue), begin(buf_a), end(buf_a), 1.f);
			auto compute_b = actions::transform(distr<class compute_b>(queue), begin(buf_a), end(buf_a), begin(buf_b), [](const float x) { return 2.f * x; });
			auto compute_c = actions::transform(master(queue), begin(buf_a), end(buf_a), begin(buf_c), [](const float x) { return 2.f - x; });
			auto compute_d = actions::transform(distr<class compute_d>(queue), begin(buf_b), end(buf_b), begin(buf_c), begin(buf_d), [](const float x, const float y) { return x + y; });
			auto reduce_d  = actions::accumulate(master(queue), begin(buf_d), end(buf_d), 0.0f, [](const float acc, const float x) { return acc + x; });

			on_master(verify(produce_a | compute_b | compute_c | compute_d | reduce_d | submit_to(queue)));
		}

	}
	catch (std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (cl::sycl::exception& e) {
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
