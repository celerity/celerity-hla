#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

#include "../src/actions.h"

//#include "../src/fusion.h"

SCENARIO("Fusing two tasks", "[fusion::simple]")
{
    celerity::distr_queue q{};

    GIVEN("Two simple transform kernels and a buffer of hundred 1s")
    {
        auto add_5 = [](int x) { return x + 5; };
        auto mul_3 = [](int x) { return x * 3; };

        constexpr auto size = 100;
        std::vector<int> src(size, 1);
        celerity::buffer<int, 1> buf_in(src.data(), {size});

        WHEN("chaining calls targeting same input/output buffers")
        {
            //celerity::buffer<int, 1> buf_out(buf_in.get_range());

            auto seq = buf_in |
                       transform<class add>(q, {}, {}, {}, add_5) |
                       transform<class mul>(q, {}, {}, {}, mul_3);

            auto buf_out = seq | submit_to(q);

            THEN("kernels get fused and the result is 18")
            {
                static_assert(size_v<decltype(seq)> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(next(begin(r)), prev(end(r))));
            }
        }
    }
}