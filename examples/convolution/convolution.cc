#include <cassert>
#include <cstdio>
#include <vector>

#define MOCK_CELERITY
#include "../../src/celerity.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "../../src/algorithm.h"

cl::sycl::float3 fmin(cl::sycl::float3 a, cl::sycl::float3 b)
{
	return { std::min(a.x(), b.x()), std::min(a.y(), b.y()), std::min(a.z(), a.z()) };
}

bool is_on_boundary(cl::sycl::range<2> range, size_t filter_size, cl::sycl::id<2> id) {
	return (id[0] < (filter_size / 2) || id[1] < (filter_size / 2) || id[0] > range[0] - (filter_size / 2) - 1 || id[1] > range[1] - (filter_size / 2) - 1);
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	std::vector<cl::sycl::float3> image_input;
	int image_width = 0, image_height = 0, image_channels = 0;
	{
		uint8_t* image_data = stbi_load(argv[1], &image_width, &image_height, &image_channels, 3);
		assert(image_data != nullptr);
		image_input.resize(image_height * image_width);
		for(auto y = 0; y < image_height; ++y) {
			for(auto x = 0; x < image_width; ++x) {
				const auto idx = y * image_width * 3 + x * 3;
				image_input[y * image_width + x] = {image_data[idx + 0] / 255.f, image_data[idx + 1] / 255.f, image_data[idx + 2] / 255.f};
			}
		}

		stbi_image_free(image_data);
	}

	constexpr int FILTER_SIZE = 8;
	constexpr float sigma = 3.f;
	constexpr float PI = 3.141592f;

	std::vector<float> gaussian_matrix(FILTER_SIZE * FILTER_SIZE);
	for(size_t j = 0; j < FILTER_SIZE; ++j) {
		for(size_t i = 0; i < FILTER_SIZE; ++i) {
			const auto x = i - (FILTER_SIZE / 2);
			const auto y = j - (FILTER_SIZE / 2);
			const auto value = std::exp(-1.f * (x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
			gaussian_matrix[j * FILTER_SIZE + i] = value;
		}
	}

	using namespace celerity;
	using namespace algorithm;
	
	distr_queue queue;

	buffer<cl::sycl::float3, 2> image_input_buf(image_input.data(), cl::sycl::range<2>(image_height, image_width));
	buffer<cl::sycl::float3, 2> image_tmp_buf(cl::sycl::range<2>(image_height, image_width));

	buffer<float, 2> gaussian_mat_buf(gaussian_matrix.data(), cl::sycl::range<2>(FILTER_SIZE, FILTER_SIZE));
	
	transform(distr<class gaussian_blur>(queue), begin(image_input_buf), end(image_input_buf), begin(gaussian_mat_buf), begin(image_tmp_buf),
		[fs = FILTER_SIZE, image_height, image_width](cl::sycl::item<2> item, chunk<cl::sycl::float3, FILTER_SIZE / 2, FILTER_SIZE / 2> in, all<float, 2> gauss)
		{
			using cl::sycl::float3;
			if (is_on_boundary(cl::sycl::range<2>(image_height, image_width), fs, item)) {
				return float3(0.f, 0.f, 0.f);
			}

			float3 sum = { 0.f, 0.f, 0.f };
			for (auto y = -(fs / 2); y < fs / 2; ++y) {
				for (auto x = -(fs / 2); x < fs / 2; ++x) {
					sum += gauss[{static_cast<size_t>(fs / 2) + y, static_cast<size_t>(fs / 2) + x}] * in[{y, x}];
				}
			}

			return sum;
		});
	
	// Do a gaussian blur
	/*queue.submit([=](handler& cgh) {
		auto in = image_input_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(FILTER_SIZE / 2, FILTER_SIZE / 2));
		auto gauss = gaussian_mat_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2, 2>());
		auto out = image_tmp_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

		cgh.parallel_for<class gaussian_blur>(cl::sycl::range<2>(image_height, image_width), [=, fs = FILTER_SIZE](cl::sycl::item<2> item) {
			using cl::sycl::float3;
			if(is_on_boundary(cl::sycl::range<2>(image_height, image_width), fs, item)) {
				out[item] = float3(0.f, 0.f, 0.f);
				return;
			}

			float3 sum = float3(0.f, 0.f, 0.f);
			for(auto y = -(fs / 2); y < fs / 2; ++y) {
				for(auto x = -(fs / 2); x < fs / 2; ++x) {
					sum += gauss[cl::sycl::id<2>(fs / 2 + y, fs / 2 + x)] * in[{item[0] + y, item[1] + x}];
				}
			}
			out[item] = sum;
		});
	});*/

	buffer<cl::sycl::float3, 2> image_output_buf(cl::sycl::range<2>(image_height, image_width));

	transform(distr<class sharpening>(queue), begin(image_tmp_buf), end(image_tmp_buf), begin(image_output_buf),
		[fs = FILTER_SIZE, image_height, image_width](cl::sycl::item<2> item, chunk<cl::sycl::float3, 1, 1> in)
		{
			using cl::sycl::float3;
			if (is_on_boundary(cl::sycl::range<2>(image_height, image_width), fs, item)) {
				return float3(0.f, 0.f, 0.f);
			}

			float3 sum = 5.f * *in;
			sum -= in[{-1, 0}];
			sum -= in[{ 1, 0}];
			sum -= in[{ 0,-1}];
			sum -= in[{ 0, 1}];
		
			return fmin(float3(1.f, 1.f, 1.f), sum);
		});

	// Now apply a sharpening kernel
	/*queue.submit([=](handler& cgh) {
		auto in = image_tmp_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
		auto out = image_output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.parallel_for<class sharpen>(cl::sycl::range<2>(image_height, image_width), [=, fs = FILTER_SIZE](cl::sycl::item<2> item) {
			using cl::sycl::float3;
			if(is_on_boundary(cl::sycl::range<2>(image_height, image_width), fs, item)) {
				out[item] = float3(0.f, 0.f, 0.f);
				return;
			}

			float3 sum = 5.f * in[item];
			sum -= in[{item[0] - 1, item[1]}];
			sum -= in[{item[0] + 1, item[1]}];
			sum -= in[{item[0], item[1] - 1}];
			sum -= in[{item[0], item[1] + 1}];
			out[item] = fmin(float3(1.f, 1.f, 1.f), sum);
		});
	});*/

	std::vector<std::array<uint8_t, 3>> image_output(image_width* image_height);
	transform(master(queue), begin(image_input_buf), end(image_input_buf), begin(image_output),
		[](cl::sycl::float3 c) -> std::array<uint8_t, 3>
		{
			return {
				static_cast<uint8_t>(static_cast<float>(c.x()) * 255.f),
				static_cast<uint8_t>(static_cast<float>(c.y()) * 255.f),
				static_cast<uint8_t>(static_cast<float>(c.z()) * 255.f)
			};
		});

	stbi_write_png("./output.png", image_width, image_height, 3, image_output.data(), 0);
	

	
	
	return EXIT_SUCCESS;
}
