#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wreturn-type"

#include "../../src/celerity_helper.h"
#include "../../src/algorithm.h"
#include "../../src/actions.h"
#include "../../src/buffer_traits.h"

auto setup_wave(cl::sycl::range<2> range, cl::sycl::float2 center, float amplitude, cl::sycl::float2 sigma)
{
	using namespace celerity::algorithm;

	return generate_n<class setup>(range,
								 [c = center, a = amplitude, s = sigma](cl::sycl::item<2> item) {
									 const auto dx = item[1] - c.x();
									 const auto dy = item[0] - c.y();
									 return a * cl::sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
								 });
}

auto zero(cl::sycl::range<2> &range)
{
	using namespace celerity::algorithm;
	return fill_n<class zero>(range, 0.f);
}

struct init_config
{
	static constexpr float a = 0.5f;
	static constexpr float b = 0.0f;
	static constexpr float c = 0.5f;
};

struct update_config
{
	static constexpr float a = 1.f;
	static constexpr float b = 1.f;
	static constexpr float c = 1.f;
};

template <typename Config, typename KernelName>
auto step(float dt, cl::sycl::float2 delta)
{
	using namespace celerity::algorithm;

	const auto step = [=](float v_up, const chunk<float, 1, 1> &v_u) {
		const float lap = (dt / delta.y()) * (dt / delta.y()) * ((v_u[{1, 0}] - *v_u) - (*v_u - v_u[{-1, 0}])) + (dt / delta.x()) * (dt / delta.x()) * ((v_u[{0, 1}] - *v_u) - (*v_u - v_u[{0, -1}]));

		return Config::a * 2 * *v_u - Config::b * v_up + Config::c * lap;
	};

	return transform<KernelName>(step);
}

auto initialize(float dt, cl::sycl::float2 delta)
{
	return step<init_config, class initialize>(dt, delta);
}

auto update(float dt, cl::sycl::float2 delta)
{
	return step<update_config, class update>(dt, delta);
}

template <typename T>
void store(celerity::distr_queue &queue, celerity::buffer<T, 2> &up, std::vector<std::vector<float>> &result_frames)
{
	using namespace celerity::algorithm;

	const auto range = up.get_range();
	std::vector<float> v(range.size());

	copy(master_blocking(queue), up, v.data());

	result_frames.emplace_back(std::move(v));
}

void write_csv(size_t N, std::vector<std::vector<float>> &result_frames)
{
	std::ofstream os;
	os.open("wave_sim_result.csv", std::ios_base::out | std::ios_base::binary);

	os << "t";
	for (size_t y = 0; y < N; ++y)
	{
		for (size_t x = 0; x < N; ++x)
		{
			os << "," << y << ":" << x;
		}
	}
	os << "\n";

	size_t i = 0;
	for (auto &frame : result_frames)
	{
		os << i++;
		for (size_t y = 0; y < N; ++y)
		{
			for (size_t x = 0; x < N; ++x)
			{
				auto v = frame[y * N + x];
				os << "," << v;
			}
		}
		os << "\n";
	}
}

struct wave_sim_config
{
	int N = 64;	// Grid size
	float T = 100; // Time at end of simulation
	float dt = 0.25f;
	float dx = 1.f;
	float dy = 1.f;

	// "Sample" a frame every X iterations
	// (0 = don't produce any output)
	unsigned output_sample_rate = 3;
};

using arg_vector = std::vector<const char *>;

template <typename ArgFn, typename Result>
bool get_cli_arg(const arg_vector &args, const arg_vector::const_iterator &it, const std::string &argstr, Result &result, ArgFn fn)
{
	if (argstr == *it)
	{
		if (it + 1 == args.cend())
		{
			throw std::runtime_error("Invalid argument");
		}
		result = static_cast<Result>(fn(*(it + 1)));
		return true;
	}
	return false;
}

int main(int argc, char *argv[])
{
	using namespace celerity::algorithm;

	// Explicitly initialize here so we can use MPI functions right away
	celerity::runtime::init(&argc, &argv);

	// Parse command line arguments
	const auto cfg = ([&]() {
		wave_sim_config result;
		const arg_vector args(argv + 1, argv + argc);
		for (auto it = args.cbegin(); it != args.cend(); ++it)
		{
			if (get_cli_arg(args, it, "-N", result.N, atoi) || get_cli_arg(args, it, "-T", result.T, atoi) || get_cli_arg(args, it, "--dt", result.dt, atof) || get_cli_arg(args, it, "--sample-rate", result.output_sample_rate, atoi))
			{
				++it;
				continue;
			}
			std::cerr << "Unknown argument: " << *it << std::endl;
		}
		return result;
	})(); // IIFE

	const auto num_steps = static_cast<int>(cfg.T / cfg.dt);
	on_master([&]() {
		if (cfg.output_sample_rate != 0 && num_steps % cfg.output_sample_rate != 0)
		{
			std::cerr << "Warning: Number of time steps (" << num_steps << ") is not a multiple of the output sample rate (wasted frames)" << std::endl;
		}
	});

	//celerity::experimental::bench::log_user_config({ {"N", std::to_string(cfg.N)}, {"T", std::to_string(cfg.T)}, {"dt", std::to_string(cfg.dt)},
	//	{"dx", std::to_string(cfg.dx)}, {"dy", std::to_string(cfg.dy)}, {"outputSampleRate", std::to_string(cfg.output_sample_rate)} });

	auto is_master = false;
	on_master([&]() { is_master = true; });

	// TODO: We could allocate the required size at the beginning
	std::vector<std::vector<float>> result_frames;
	{
		celerity::distr_queue queue;
		cl::sycl::range buf_range = cl::sycl::range<2>(cfg.N, cfg.N);

		//celerity::algorithm::actions::sync();
		MPI_Barrier(MPI_COMM_WORLD);

		celerity::experimental::bench::begin("main program");

		auto u = setup_wave(buf_range, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f}) |
				 submit_to(queue);

		auto up = fill_n<class zero>(buf_range, 0.f) |
				  initialize(cfg.dt, {cfg.dx, cfg.dy}) << u |
				  submit_to(queue);

		// We need to rotate buffers. Since we cannot swap them directly, we use pointers instead.
		// TODO: Make buffers swappable
		auto up_ref = &up;
		auto u_ref = &u;

		// Store initial state
		if (cfg.output_sample_rate > 0)
		{
			store(queue, *u_ref, result_frames);
		}

		auto t = 0.0;
		size_t i = 0;
		while (t < cfg.T)
		{
			*up_ref | update(cfg.dt, {cfg.dx, cfg.dy}) << *u_ref | *up_ref | submit_to(queue);

			if (cfg.output_sample_rate != 0 && ++i % cfg.output_sample_rate == 0)
			{
				store(queue, *up_ref, result_frames);
			}

			std::swap(u_ref, up_ref);
			t += cfg.dt;
		}
	}

	if (is_master)
		if (cfg.output_sample_rate > 0)
		{
			std::cout << "writing output..." << std::endl;

			// TODO: Consider writing results to disk as they're coming in, instead of just at the end
			write_csv(cfg.N, result_frames);
		}

	return EXIT_SUCCESS;
}
