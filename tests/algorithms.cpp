#define CATCH_CONFIG_MAIN
#define CELERITY_TEST
#include "../src/catch/catch.hpp"
#include "../src/sycl.h"
#include "../src/algorithm.h"

using namespace celerity;

struct GlobalSetupAndTeardown : Catch::TestEventListenerBase
{
    using TestEventListenerBase::TestEventListenerBase;
    void testRunStarting(const Catch::TestRunInfo &) override { celerity::detail::runtime::enable_test_mode(); }
    void testCaseEnded(const Catch::TestCaseStats &) override
    {
        if (celerity::detail::runtime::is_initialized())
        {
            celerity::detail::runtime::teardown();
        }
    }
};

CATCH_REGISTER_LISTENER(GlobalSetupAndTeardown)

template <int Rank, typename T>
std::vector<T> copy_to_host(distr_queue &q, celerity::buffer<T, Rank> &src)
{
    std::vector<int> dst(src.get_range().size());
    algorithm::copy(algorithm::master_blocking(q), src, dst.data());
    return dst;
}

SCENARIO("filling a buffer", "[celerity::algorithm]")
{
    GIVEN("A one-dimensional buffer of hundred elements")
    {
        constexpr auto size = 100;

        distr_queue q;
        buffer<int, 1> buf(cl::sycl::range<1>{size});

        WHEN("filling with 1s")
        {
            algorithm::fill<class fill_ones>(q, buf, 1);

            THEN("the buffer contains only 1s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 1; }));
            }
        }

        WHEN("filling with 2s using piping syntax")
        {
            buf | algorithm::fill<class fill_twos>(q, {}, 2) | q;

            THEN("the buffer contains only 2s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 2; }));
            }
        }
    }
}