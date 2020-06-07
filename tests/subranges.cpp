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

    GIVEN("A buffer of 100 ones and a transform kernel which adds 5")
    {
        constexpr auto size = 100;
        buffer<int, 1> in_buf{size};
        fill<class _0>(q, in_buf, 1);

        auto t0 = transform<class _1>([](int x) { return x + 5; });

        WHEN("skipping the first element")
        {
            auto r0 = skip<1>({1});
            auto buf_out = in_buf | r0 | t0 | submit_to(q);

            THEN("the first element is zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());
                REQUIRE(r.front() == 0);
                REQUIRE(elements_equal_to<6>(next(begin(r)), end(r)));
            }
        }

        WHEN("taking 99 elements")
        {
            auto r0 = take<1>({99});
            buffer<int, 1> buf_out{size};

            in_buf | r0 | t0 | buf_out | submit_to(q);

            THEN("the last element is zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());
                REQUIRE(r.back() == 0);
                REQUIRE(elements_equal_to<6>(begin(r), prev(end(r))));
            }
        }

        WHEN("skipping 33 and taking 33 elements")
        {
            auto r0 = skip<1>({33});
            auto r1 = take<1>({33});
            buffer<int, 1> buf_out{size};

            in_buf | r0 | r1 | t0 | buf_out | submit_to(q);

            THEN("the first 33 and last 34 are zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());
                REQUIRE(elements_equal_to<0>(begin(r), begin(r) + 33));
                REQUIRE(elements_equal_to<6>(begin(r) + 33, begin(r) + 66));
                REQUIRE(elements_equal_to<0>(begin(r) + 66, end(r)));
            }
        }
    }
}