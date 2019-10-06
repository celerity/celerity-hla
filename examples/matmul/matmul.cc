//#define MOCK_CELERITY
#include "../../src/algorithm.h"
#include "../../src/actions.h"

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

	std::vector<float> mat_a(MAT_SIZE * MAT_SIZE);
	std::vector<float> mat_b(MAT_SIZE * MAT_SIZE);

	// Initialize matrices a and b to the identity
	for (size_t i = 0; i < MAT_SIZE; ++i)
	{
		for (size_t j = 0; j < MAT_SIZE; ++j)
		{
			mat_a[i * MAT_SIZE + j] = i == j;
			mat_b[i * MAT_SIZE + j] = i == j;
		}
	}

	try
	{
		celerity::distr_queue queue;
		celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>{MAT_SIZE, MAT_SIZE});
		celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>{MAT_SIZE, MAT_SIZE});
		celerity::buffer<float, 2> mat_c_buf(cl::sycl::range<2>{MAT_SIZE, MAT_SIZE});

		auto multiply = [](slice<float, 1> a, slice<float, 0> b) {
			auto sum = 0.f;

			for (auto k = 0; k < MAT_SIZE; ++k)
			{
				const auto a_ik = a[k];
				const auto b_kj = b[k];
				sum += a_ik * b_kj;
			}

			return sum;
		};

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		transform(algorithm::distr<class mul_ab>(queue), begin(mat_a_buf), end(mat_a_buf), begin(mat_b_buf), begin(mat_c_buf), multiply);
		transform(algorithm::distr<class mul_bc>(queue), begin(mat_b_buf), end(mat_b_buf), begin(mat_c_buf), begin(mat_a_buf), multiply);

		for_each(algorithm::master_blocking(queue), begin(mat_a_buf), end(mat_a_buf),
				 [&verification_passed](cl::sycl::item<2> item, float x) {
					 const float correct_value = item[0] == item[1];

					 if (x == correct_value)
						 return;

					 fprintf(stderr, "VERIFICATION FAILED for element %llu,%llu: %f != %f\n", item[0], item[1], x, correct_value);
					 verification_passed = false;
				 });

		actions::on_master([&]() {
			if (verification_passed)
			{
				printf("VERIFICATION PASSED!\n");
			}
		});
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
