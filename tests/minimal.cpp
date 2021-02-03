#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

void benchmark(size_t size)
{
}

TEST_CASE("transforming a buffer", "[celerity::algorithm]")
{
    distr_queue q;

    {
        constexpr auto size = 100;
        std::vector<int> v(size, 1); 
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});

        buffer<int, 1> buf_out(buf.get_range());

        transform(distr<class _28>(q), begin(buf), end(buf), begin(buf_out),
                  [](const algorithm::chunk<int, 1> &x) { return 3 * (*x); });

        const auto r = copy_to_host(q, buf_out);
        CHECK(elements_equal_to<3>(r));
    };

    BENCHMARK_ADVANCED("size == 100")
    (Catch::Benchmark::Chronometer meter)
    {
        constexpr auto size = 100;
        std::vector<int> v(size, 1);
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});
        buffer<int, 1> buf_out(buf.get_range());

        meter.measure([&]() {
            transform(distr<class _44>(q), begin(buf), end(buf), begin(buf_out),
                      [](const algorithm::chunk<int, 1> &x) { return 3 * (*x); });
            q.slow_full_sync();
        });
    };

    BENCHMARK_ADVANCED("size == 1000")
    (Catch::Benchmark::Chronometer meter)
    {
        constexpr auto size = 1000;
        std::vector<int> v(size, 1);
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});
        buffer<int, 1> buf_out(buf.get_range());

        meter.measure([&]() {
            transform(distr<class _59>(q), begin(buf), end(buf), begin(buf_out),
                      [](const algorithm::chunk<int, 1> &x) { return 3 * (*x); });
            q.slow_full_sync();
        });
    };

    BENCHMARK_ADVANCED("size == 100 000")
    (Catch::Benchmark::Chronometer meter)
    {
        constexpr auto size = 100000;
        std::vector<int> v(size, 1);
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});
        buffer<int, 1> buf_out(buf.get_range());

        meter.measure([&]() {
            transform(distr<class _74>(q), begin(buf), end(buf), begin(buf_out),
                      [](const algorithm::chunk<int, 1> &x) { return 3 * (*x); });
            q.slow_full_sync();
        });
    };

    BENCHMARK_ADVANCED("size == 1 000 000")
    (Catch::Benchmark::Chronometer meter)
    {
        constexpr auto size = 1000000;
        std::vector<int> v(size, 1);
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});
        buffer<int, 1> buf_out(buf.get_range());

        meter.measure([&]() {
            transform(distr<class _89>(q), begin(buf), end(buf), begin(buf_out),
                      [](const algorithm::chunk<int, 1> &x) { return 3 * (*x); });
            q.slow_full_sync();
        });
    };
}