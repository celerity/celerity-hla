#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;

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

        WHEN("filling with numbers 1 to 100")
        {
            algorithm::generate<class iota>(q, buf, [](cl::sycl::item<1> i) { return i.get_linear_id() + 1; });

            THEN("the buffer contains numbers 1 to 100")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::accumulate(begin(r), end(r), 0, std::plus<int>()) == size * (size + 1) / 2);
            }
        }
    }
}