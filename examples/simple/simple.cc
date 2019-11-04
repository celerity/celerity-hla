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

		/*queue.submit([=](celerity::handler& cgh) {
			auto dw_a = buf_a.get_access<cl::sycl::access::mode::discard_write>(cgh, [](celerity::chunk<1> chnk) -> celerity::subrange<1> {
				celerity::subrange<1> sr(chnk);
				// Write the opposite subrange
				// This is useful to demonstrate that the nodes are assigned to
				// chunks somewhat intelligently in order to minimize buffer
				// transfers. Remove this line and the node assignment in the
				// command graph should be flipped.
				sr.offset = cl::sycl::id<1>(chnk.global_size) - chnk.offset - cl::sycl::id<1>(chnk.range);
				return sr;
			});

			cgh.parallel_for<class produce_a>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) { dw_a[DEMO_DATA_SIZE - 1 - item[0]] = 1.f; });
		});*/ 

		//algorithm::fill<class produce_a_1>(queue, buf_a, 1.f) |
		//	algorithm::transform<class compute_b_1>(queue, {}, buf_b, [](float x) { return 2.f * x; });
 
		buf_a |  
			algorithm::fill<class produce_a>(queue, {}, 1.f) |
			algorithm::transform<class compute_c>(queue, {}, buf_c,  [](float x) { return 2.f - x; }) | 
			algorithm::transform<class compute_b>(queue, buf_a, buf_b, [](float x) { return 2.f * x; }) | // parallel 
			algorithm::transform<class compute_d>(queue, {}, {}, buf_d, std::plus<float>{}) << buf_c;
			
	/*queue.submit([=](celerity::handler& cgh) {
			auto r_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, [](celerity::chunk<1> chnk) -> celerity::subrange<1> {
				celerity::subrange<1> sr(chnk);
				// Add some overlap so we can generate pull commands
				// NOTE: JUST A DEMO. NOT HONORED IN KERNEL.
				if(chnk.offset[0] > 10) { sr.offset -= 10; }
				sr.range += 20;
				return sr;
			});

			auto dw_b = buf_b.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
			cgh.parallel_for<class compute_b>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) { dw_b[item] = r_a[item] * 2.f; });
		});

		*/

#define COMPUTE_C_ON_MASTER 0
#if COMPUTE_C_ON_MASTER
		/*
		queue.with_master_access([=](celerity::handler& cgh) {
			auto r_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(DEMO_DATA_SIZE));
			auto dw_c = buf_c.get_access<cl::sycl::access::mode::discard_write>(cgh, cl::sycl::range<1>(DEMO_DATA_SIZE));

			cgh.run([=]() {
				for(int i = 0; i < DEMO_DATA_SIZE; ++i) {
					dw_c[i] = 2.f - r_a[i];
				}
			});
		});
		*/

		transform(algorithm::master(queue), begin(buf_a), end(buf_a), begin(buf_c), [](float x) { return 2.f - x; });

#else
		/*
		queue.submit([=](celerity::handler& cgh) {
			auto r_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto dw_c = buf_c.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
			cgh.parallel_for<class compute_c>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) { dw_c[item] = 2.f - r_a[item]; });
		});
		*/

		//algorithm::transform<class compute_c>(queue, buf_a, buf_c, [](float x) { return 2.f - x; });
#endif
		/*
		queue.submit([=](celerity::handler& cgh) {
			auto r_b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto r_c = buf_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto dw_d = buf_d.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
			cgh.parallel_for<class compute_d>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) { dw_d[item] = r_b[item] + r_c[item]; });
		});

		*/

		//algorithm::transform<class compute_d>(queue, buf_b, buf_c, buf_d, std::plus<float>{});

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

		//const auto sum = accumulate(algorithm::master_blocking(queue), begin(buf_d), end(buf_d), 0.0f, [](float acc, float x) { return acc + x; });

		// OR
		//float sum = accumulate(algorithm::master_blocking(queue), begin(buf_d), end(buf_d), 0.0f, std::plus<float>{});
		//                                   ^^^^^^^^^^^^^^^^^^^^^^

		/*algorithm::on_master([&]() {
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
		});*/
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
