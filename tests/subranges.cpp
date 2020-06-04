#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include "../include/sequencing.h"
#include "../include/actions.h"
#include "../include/fusion_helper.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;
using namespace celerity::algorithm::traits;
using namespace celerity::algorithm::util;
using celerity::algorithm::chunk;

SCENARIO("Subranges", "[subranges::simple]")
{
    celerity::distr_queue q{};

    GIVEN("Two simple transform kernels and a buffer of hundred 1s")
    {
        auto add_5 = [](int x) { return x + 5; };

        constexpr auto size = 100;
        std::vector<int> src(size, 1);
        buffer<int, 1> in_buf{size};
        fill<class _0>(q, in_buf, 1);

        WHEN("chaining calls")
        {
            auto r0 = skip<1>({1});
            auto t0 = transform<class _1>(add_5);

            auto seq = in_buf | r0 | t0;

            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.front() == 0);
                REQUIRE(elements_equal_to<6>(next(begin(r)), end(r)));
            }
        }
    }
}