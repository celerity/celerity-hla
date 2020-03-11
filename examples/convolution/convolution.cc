#include <cassert>
#include <cstdio>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wreturn-type"

#include "../../src/algorithm.h"
#include "../../src/buffer_traits.h"

constexpr int FILTER_SIZE = 16;
constexpr float sigma = 3.f;
constexpr float PI = 3.141592f;

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	std::vector<cl::sycl::float3> image_input;
	int image_width = 0, image_height = 0, image_channels = 0;

	{
		uint8_t *image_data = stbi_load(argv[1], &image_width, &image_height, &image_channels, 3);
		assert(image_data != nullptr);
		image_input.resize(image_height * image_width);
		for (auto y = 0; y < image_height; ++y)
		{
			for (auto x = 0; x < image_width; ++x)
			{
				const auto idx = y * image_width * 3 + x * 3;
				image_input[y * image_width + x] = {image_data[idx + 0] / 255.f, image_data[idx + 1] / 255.f, image_data[idx + 2] / 255.f};
			}
		}

		stbi_image_free(image_data);
	}

	using namespace celerity;
	using namespace algorithm;

	using f = celerity::algorithm::buffer_traits<float, 2>;
	using f3 = celerity::algorithm::buffer_traits<cl::sycl::float3, 2>;

	const auto generate_gaussian_mat = [](cl::sycl::item<2> item) {
		const auto x = item.get_id(1) - (FILTER_SIZE / 2);
		const auto y = item.get_id(0) - (FILTER_SIZE / 2);

		return cl::sycl::exp(-1.f * (x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
	};

	const auto gaussian_blur = [fs = FILTER_SIZE](const f3::chunk<FILTER_SIZE, FILTER_SIZE> &in, const f::all &gauss) {
		using cl::sycl::float3;

		return in.discern(float3(0.f, 0.f, 0.f),
						  [&]() {
							  float3 sum = {0.f, 0.f, 0.f};
							  for (auto y = -(fs / 2); y < fs / 2; ++y)
							  {
								  for (auto x = -(fs / 2); x < fs / 2; ++x)
								  {
									  sum += gauss[{static_cast<size_t>(fs / 2 + y), static_cast<size_t>(fs / 2 + x)}] * in[{y, x}];
								  }
							  }

							  return sum;
						  });
	};

	const auto sharpening = [](f3::chunk<3, 3> in) {
		using cl::sycl::float3;

		return in.discern(float3(0.f, 0.f, 0.f),
						  [&]() {
							  float3 sum = 5.f * (*in);
							  sum -= in[{-1, 0}];
							  sum -= in[{1, 0}];
							  sum -= in[{0, -1}];
							  sum -= in[{0, 1}];

							  return float3{
								  std::max(0.f, std::min(1.f, sum.x())),
								  std::max(0.f, std::min(1.f, sum.y())),
								  std::max(0.f, std::min(1.f, sum.z()))};

							  //return fmax(float3(0, 0, 0), fmin(float3(1.f, 1.f, 1.f), sum)); HIP ERROR 77
						  });
	};

	const auto convert_to_uint8 = [](cl::sycl::float3 c) -> uchar3 {
		return {
			static_cast<u_char>(c.x() * 255.f),
			static_cast<u_char>(c.y() * 255.f),
			static_cast<u_char>(c.z() * 255.f)};
	};

	distr_queue queue;
	cl::sycl::range<2> image_range(image_height, image_width);
	cl::sycl::range<2> gauss_mat_range(FILTER_SIZE, FILTER_SIZE);

	MPI_Barrier(MPI_COMM_WORLD);
	celerity::experimental::bench::begin("main program");

	auto out_buf = make_buffer(image_input.data(), image_range) |
				   transform<class gaussian_blur>(gaussian_blur) << generate<class gen_gauss>(gauss_mat_range, generate_gaussian_mat) |
				   transform<class sharpening>(sharpening) |
				   transform<class to_uchar>(convert_to_uint8) |
				   submit_to(queue);

	// auto seq = image_input_buf |
	// 		   (transform<class gaussian_blur>(gaussian_blur) << gaussian_mat_buf) |
	// 		   transform<class sharpening>(sharpening) |
	// 		   transform<class to_uchar>(convert_to_uint8);

	// static_assert(size_v<decltype(terminate(seq))> == 3);
	// static_assert(size_v<decltype(fuse(terminate(seq)))> == 2);

	// auto out_buf = seq | submit_to(queue);

	std::vector<std::array<uint8_t, 3>> image_output(image_width * image_height);
	copy(master_blocking(queue), begin(out_buf), end(out_buf), image_output.data());

	stbi_write_png("./output.png", image_width, image_height, 3, image_output.data(), 0);

	return EXIT_SUCCESS;
}
