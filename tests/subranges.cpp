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

    GIVEN("Two buffers of 100 ones and a transform kernel which sums the elements")
    {
        constexpr auto size = 100;
        buffer<int, 1> in_buf_a{size};
        buffer<int, 1> in_buf_b{size};
        fill<class _4>(q, in_buf_a, 1);
        fill<class _5>(q, in_buf_b, 2);

        auto t0 = transform<class _6>(std::plus<int>{});

        WHEN("skipping the first element of both buffers")
        {
            auto r0 = skip<1>({1});
            auto buf_out = in_buf_a | r0 | t0 << (in_buf_b | r0) | submit_to(q);

            THEN("the first element is zero and the rest is 3")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf_a.get_range().size());
                REQUIRE(r.front() == 0);
                REQUIRE(elements_equal_to<3>(next(begin(r)), end(r)));
            }
        }

        WHEN("skipping the first 25 elements and taking the next 50 elements of both buffers")
        {
            auto r0 = skip<1>({25});
            auto r1 = take<1>({50});
            buffer<int, 1> buf_out{size};

            in_buf_a | r0 | r1 | t0 << (in_buf_b | r0 | r1) | buf_out | submit_to(q);

            THEN("the first and last 25 elements are zero and the rest is 3")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf_a.get_range().size());
                REQUIRE(elements_equal_to<0>(begin(r), begin(r) + 25));
                REQUIRE(elements_equal_to<3>(begin(r) + 25, begin(r) + 75));
                REQUIRE(elements_equal_to<0>(begin(r) + 75, end(r)));
            }
        }
    }

    GIVEN("A 2d buffer of 100 ones and a transform kernel which adds 5")
    {
        constexpr auto size = 10;
        buffer<int, 2> in_buf{{size, size}};
        fill<class _2>(q, in_buf, 1);

        auto t0 = transform<class _3>([](int x) { return x + 5; });

        WHEN("skipping the first element")
        {
            auto r0 = skip<2>({0, 1});
            auto buf_out = in_buf | r0 | t0 | submit_to(q);

            THEN("the first element of every row is zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());

                for (int i = 0; i < size; ++i)
                {
                    const auto row_begin = begin(r) + 10 * i;
                    const auto row_end = begin(r) + 10 * (i + 1);

                    REQUIRE(*row_begin == 0);
                    REQUIRE(elements_equal_to<6>(row_begin + 1, row_end));
                }
            }
        }

        WHEN("taking 90 elements")
        {
            auto r0 = take<2>({9, 10});
            buffer<int, 2> buf_out{{size, size}};

            in_buf | r0 | t0 | buf_out | submit_to(q);

            THEN("the last ten elements are zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());
                REQUIRE(elements_equal_to<6>(begin(r), begin(r) + 90));
                REQUIRE(elements_equal_to<0>(begin(r) + 90, end(r)));
            }
        }

        WHEN("taking the first five elements of every row")
        {
            auto r0 = take<2>({10, 5});
            buffer<int, 2> buf_out{{size, size}};

            in_buf | r0 | t0 | buf_out | submit_to(q);

            THEN("the last five elements are zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());

                for (int i = 0; i < size; ++i)
                {
                    const auto row_begin = begin(r) + 10 * i;
                    const auto row_end = begin(r) + 10 * (i + 1);

                    REQUIRE(elements_equal_to<6>(row_begin, row_begin + 5));
                    REQUIRE(elements_equal_to<0>(row_begin + 5, row_end));
                }
            }
        }

        WHEN("taking the last five elements of every row")
        {
            auto r0 = skip<2>({0, 5});
            auto r1 = take<2>({10, 5});
            buffer<int, 2> buf_out{{size, size}};

            in_buf | r0 | r1 | t0 | buf_out | submit_to(q);

            THEN("the first five elements are zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());

                for (int i = 0; i < size; ++i)
                {
                    const auto row_begin = begin(r) + 10 * i;
                    const auto row_end = begin(r) + 10 * (i + 1);

                    REQUIRE(elements_equal_to<0>(row_begin, row_begin + 5));
                    REQUIRE(elements_equal_to<6>(row_begin + 5, row_end));
                }
            }
        }

        WHEN("skipping 30 and taking 30 elements")
        {
            auto r0 = skip<2>({3, 0});
            auto r1 = take<2>({3, 10});
            buffer<int, 2> buf_out{{size, size}};

            in_buf | r0 | r1 | t0 | buf_out | submit_to(q);

            THEN("the first 30 and last 40 are zero and the rest is 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(r.size() == in_buf.get_range().size());
                REQUIRE(elements_equal_to<0>(begin(r), begin(r) + 30));
                REQUIRE(elements_equal_to<6>(begin(r) + 30, begin(r) + 60));
                REQUIRE(elements_equal_to<0>(begin(r) + 60, end(r)));
            }
        }
    }
}