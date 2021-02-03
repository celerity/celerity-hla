#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include <numeric>

#include "../include/experimental/algorithms/fill.h"
#include "../include/experimental/algorithms/generate.h"
#include "../include/experimental/algorithms/transform.h"
#include "../include/experimental/algorithms/zip.h"

using namespace celerity;
using namespace celerity::hla::experimental;

template <auto N>
constexpr inline auto sum_one_to_n = N *(N + 1) / 2;

SCENARIO("filling a buffer", "[celerity::hla]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of hundred elements")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf(cl::sycl::range<1>{size});

        WHEN("filling with 1s")
        {
            fill<class fill_ones>(q, buf, 1);

            THEN("the buffer contains only 1s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(elements_equal_to<1>(r));
            }
        }

        WHEN("filling with numbers 1 to 100")
        {
            generate<class iota>(q, buf, [](cl::sycl::item<1> i) { return i.get_linear_id() + 1; });

            THEN("the buffer contains numbers 1 to 100")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::accumulate(begin(r), end(r), 0, std::plus<int>()) == sum_one_to_n<size>);
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 elements")
    {
        constexpr auto rank = 10;

        buffer<int, 2> buf({rank, rank});

        WHEN("filling with 1s")
        {
            fill<class fill_ones_10x10>(q, buf, 1);

            THEN("the buffer contains only 1s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(elements_equal_to<1>(r));
            }
        }

        WHEN("filling with numbers 1 to 100")
        {
            generate<class iota_10x10>(q, buf, [](cl::sycl::item<2> i) { return i.get_linear_id() + 1; });

            THEN("the buffer contains numbers 1 to 100")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::accumulate(begin(r), end(r), 0, std::plus<int>()) == sum_one_to_n<rank * rank>);
            }
        }
    }

    GIVEN("A three-dimensional buffer of 10x10x10 elements")
    {
        constexpr auto rank = 10;

        buffer<int, 3> buf({rank, rank, rank});

        WHEN("filling with 1s")
        {
            fill<class fill_ones_10x10x10>(q, buf, 1);

            THEN("the buffer contains only 1s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(elements_equal_to<1>(r));
            }
        }

        WHEN("filling with numbers 1 to 100")
        {
            generate<class iota_10x10x10>(q, buf, [](cl::sycl::item<3> i) { return i.get_linear_id() + 1; });

            THEN("the buffer contains numbers 1 to 100")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::accumulate(begin(r), end(r), 0, std::plus<int>()) == sum_one_to_n<rank * rank * rank>);
            }
        }
    }
}

SCENARIO("transforming a buffer (experimental)", "[celerity::hla::experimental]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf(cl::sycl::range<1>{size});
        fill<class _340>(q, buf, 1);

        WHEN("tripling all elements into another buffer")
        {
            using namespace celerity::hla::experimental;

            buffer<int, 1> buf_out(buf.get_range());

            hla::experimental::transform<class _346>(q, begin(buf), end(buf), begin(buf_out), [](int x) { return 3 * x; });

            THEN("every element is 3 in the target buffer")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<3>(r));
            }
        }
    }

    GIVEN("Two one-dimensional buffers of a hundred 1s and a hundred 4s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf_a(cl::sycl::range<1>{size});
        buffer<int, 1> buf_b(cl::sycl::range<1>{size});

        fill<class fill_1_a>(q, buf_a, 1);
        fill<class fill_4_b>(q, buf_b, 4);

        WHEN("adding buffers")
        {
            buffer<int, 1> buf_c(buf_a.get_range());

            hla::experimental::zip<class _376>(q, buf_a, buf_b, buf_c, std::plus<int>{});

            THEN("every element is 5")
            {
                const auto r = copy_to_host(q, buf_c);
                REQUIRE(elements_equal_to<5>(r));
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 1s")
    {
        constexpr auto rank = 10;

        buffer<int, 2> buf({rank, rank});

        fill<class fill_10x10>(q, buf, 1);

        WHEN("quadrupling all elements into another buffer")
        {
            buffer<int, 2> buf_out(buf.get_range());

            hla::experimental::transform<class mul4>(q, buf, buf_out, [](int x) { return 4 * x; });

            THEN("every element is 4 in the target buffer")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<4>(r));
            }
        }
    }

    GIVEN("Two two-dimensional buffers of 10x10 1s and a 10x10 4s")
    {
        constexpr auto rank = 10;

        buffer<int, 2> buf_a({rank, rank});
        buffer<int, 2> buf_b({rank, rank});

        fill<class fill_1_a_10x10>(q, buf_a, 1);
        fill<class fill_4_b_10x10>(q, buf_b, 4);

        WHEN("adding buffers")
        {
            buffer<int, 2> buf_c(buf_a.get_range());

            hla::experimental::zip<class add_ab_10x10>(q, buf_a, buf_b, buf_c, std::plus<int>{});

            THEN("every element is 5")
            {
                const auto r = copy_to_host(q, buf_c);
                REQUIRE(elements_equal_to<5>(r));
            }
        }
    }

    GIVEN("A three-dimensional buffer of 10x10x10 1s")
    {
        constexpr auto rank = 10;

        buffer<int, 3> buf({rank, rank, rank});

        fill<class fill_ab_10x10x10>(q, buf, 1);

        WHEN("quintupling all elements into another buffer")
        {
            buffer<int, 3> buf_out(buf.get_range());

            hla::experimental::transform<class mul5>(q, buf, buf_out, [](int x) { return 5 * x; });

            THEN("every element is 5 in the target buffer")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<5>(r));
            }
        }
    }

    GIVEN("Two three-dimensional buffers of 10x10x10 1s and a 10x10x10 4s")
    {
        constexpr auto rank = 10;

        buffer<int, 3> buf_a({rank, rank, rank});
        buffer<int, 3> buf_b({rank, rank, rank});

        fill<class fill_1_a_10x10x10>(q, buf_a, 1);
        fill<class fill_4_b_10x10x10>(q, buf_b, 4);

        WHEN("adding buffers")
        {
            buffer<int, 3> buf_c(buf_a.get_range());

            hla::experimental::zip<class add_ab_10x10x10>(q, buf_a, buf_b, buf_c, std::plus<int>{});

            THEN("every element is 5")
            {
                const auto r = copy_to_host(q, buf_c);
                const auto a = copy_to_host(q, buf_a);
                const auto b = copy_to_host(q, buf_b);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<5>(r));
            }
        }
    }
}
