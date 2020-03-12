#include <cassert>
#include <cstdio>
#include <vector>
#include <numeric>

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

std::vector<cl::sycl::float3> load_image(std::string filename, int &width, int &height, int &channels)
{
	std::vector<cl::sycl::float3> image_input;

	uint8_t *image_data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
	assert(image_data != nullptr);

	image_input.resize(height * width);

	for (auto y = 0; y < height; ++y)
	{
		for (auto x = 0; x < width; ++x)
		{
			const auto idx = y * width * 3 + x * 3;
			image_input[y * width + x] = {image_data[idx + 0] / 255.f, image_data[idx + 1] / 255.f, image_data[idx + 2] / 255.f};
		}
	}

	stbi_image_free(image_data);

	return image_input;
}

namespace kernels
{

using f = celerity::algorithm::buffer_traits<float, 2>;
using f3 = celerity::algorithm::buffer_traits<cl::sycl::float3, 2>;

constexpr auto gen_gauss = [](cl::sycl::item<2> item) {
	const auto x = item.get_id(1) - (FILTER_SIZE / 2);
	const auto y = item.get_id(0) - (FILTER_SIZE / 2);

	return cl::sycl::exp(-1.f * (x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
};

constexpr auto blur = [](const f3::chunk<FILTER_SIZE, FILTER_SIZE> &in, const f::all &gauss) {
	return in.discern(cl::sycl::float3{},
					  [&]() { return std::inner_product(begin(in), end(in), begin(gauss), cl::sycl::float3{}); });
};

constexpr auto sharpen = [](f3::chunk<3, 3> in) {
	constexpr std::array<int, 9> weights = {
		0, -1, 0,
		-1, 0, -1,
		0, -1, 0};

	return in.discern(cl::sycl::float3{},
					  [&]() { return std::inner_product(begin(in), end(in), begin(weights), 5.f * (*in)); });
};

constexpr auto delimit = [](cl::sycl::float3 v) -> cl::sycl::float3 {
	return {
		std::max(0.f, std::min(1.f, v.x())),
		std::max(0.f, std::min(1.f, v.y())),
		std::max(0.f, std::min(1.f, v.z()))};
};

constexpr auto to_uint8 = [](cl::sycl::float3 c) -> uchar3 {
	return {
		static_cast<u_char>(c.x() * 255.f),
		static_cast<u_char>(c.y() * 255.f),
		static_cast<u_char>(c.z() * 255.f)};
};

} // namespace kernels

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s <image file>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int image_width = 0, image_height = 0, image_channels = 0;
	auto image_input = load_image(argv[1], image_width, image_height, image_channels);

	using namespace celerity;
	using namespace algorithm;

	distr_queue queue;
	cl::sycl::range<2> image_range(image_height, image_width);
	cl::sycl::range<2> gauss_range(FILTER_SIZE, FILTER_SIZE);

	auto out_buf = make_buffer(image_input.data(), image_range) |
				   transform<class _1>(kernels::blur) << generate<class _2>(gauss_range, kernels::gen_gauss) |
				   transform<class _3>(kernels::sharpen) |
				   transform<class _4>(kernels::delimit) |
				   transform<class _5>(kernels::to_uint8) |
				   submit_to(queue);

	// auto seq = make_buffer(image_input.data(), image_range) |
	// 		   transform<class _1>(kernels::blur) << generate<class _2>(gauss_range, kernels::gen_gauss) |
	// 		   transform<class _3>(kernels::sharpen) |
	// 		   transform<class _4>(kernels::delimit) |
	// 		   transform<class _5>(kernels::to_uint8);

	// static_assert(size_v<decltype(terminate(seq))> == 4);
	// static_assert(size_v<decltype(fuse(terminate(seq)))> == 2);

	// auto out_buf = seq | submit_to(queue);

	std::vector<std::array<uint8_t, 3>> image_output(image_width * image_height);
	copy(master_blocking(queue), begin(out_buf), end(out_buf), image_output.data());
	stbi_write_png("./output.png", image_width, image_height, image_channels, image_output.data(), 0);

	return EXIT_SUCCESS;
}
