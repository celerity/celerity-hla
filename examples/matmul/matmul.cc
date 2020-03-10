#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wreturn-type"

#include "../../src/algorithm.h"
#include "../../src/actions.h"
#include "../../src/buffer_traits.h"

constexpr auto MAT_SIZE = 1024;

using namespace celerity;
using namespace algorithm;

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

		auto gen_a = [](cl::sycl::item<2> item) {
			return static_cast<float>(item.get_id(0) == item.get_id(1));
		};

		auto gen_b = [gen_a](cl::sycl::item<2> item) {
			return gen_a(item) * 2;
		};

		auto multiply = [](const t::slice<1> &a, const t::slice<0> &b) {
			auto sum = 0.f;

			for (auto k = 0; k < MAT_SIZE; ++k)
			{
				const auto a_ik = a[k];
				const auto b_kj = b[k];
				sum += a_ik * b_kj;
			}

			return sum;
		};

		const auto mat_range = cl::sycl::range<2>{MAT_SIZE, MAT_SIZE};

		auto mat_b =
			generate<class gen_b>(mat_range, gen_b) |
			submit_to(queue);

		// auto out_buf =
		// 	generate<class gen_a>(mat_range, gen_a) |
		// 	transform<class mul_a_b>(multiply) << mat_b |
		// 	transform<class mul_ab_b>(multiply) << mat_b |
		// 	submit_to(queue);

		auto seq = generate<class gen_a>(cl::sycl::range<2>{MAT_SIZE, MAT_SIZE}, gen_a) |
				   transform<class mul_a_b>(multiply) << mat_b |
				   transform<class mul_ab_b>(multiply) << mat_b;

		auto fused = fuse(terminate(seq));

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		auto out_buf = std::get<2>(fused(queue));

		// static_assert(size_v<decltype(terminate(seq))> == 3);
		// static_assert(size_v<decltype(fuse(terminate(seq)))> == 3);

		master_task(algorithm::master(queue), [=, &verification_passed](auto &cgh) {
			auto r_d = out_buf.get_access<cl::sycl::access::mode::read>(cgh, out_buf.get_range());

			return [=, &verification_passed]() {
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
