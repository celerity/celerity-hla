#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

#include "../include/experimental/algorithms/transform.h"
#include "../include/experimental/algorithms/zip.h"

#include "../include/experimental/kernel_traits.h"

TEST_CASE("traits")
{
    namespace h = celerity::hla::experimental;

    {
        const auto k = [](h::AnyBlock auto x) { x.configure({1}); return 3 * (*x); };

        static_assert(std::is_same_v<typename h::kernel_traits<decltype(k), h::kernel_input<int, 1>>::result_type, int>);
    } // namespace celerity::hla::experimental;

    {
        const auto k = [](h::AnyBlock auto x, h::AnyBlock auto y) { x.configure({1}); return 3 * (*x) + *y; };

        static_assert(std::is_same_v<typename h::kernel_traits<decltype(k), h::kernel_input<int, 1>, h::kernel_input<double, 1>>::result_type, double>);
    }

    {
        const auto k = [](h::AnyBlock auto x) { x.configure({1}); return 3 * (*x); };

        static_assert(std::is_same_v<typename h::kernel_traits<decltype(k), h::kernel_input<int, 1>>::result_type, int>);
    }

    {
        const auto k = [](h::AnyBlock auto x, h::AnyBlock auto y) { x.configure({1}); return 3 * (*x) + *y; };

        static_assert(std::is_same_v<h::kernel_traits<decltype(k), h::kernel_input<int, 1>, h::kernel_input<double, 1>>::result_type, double>);
    }
}

TEST_CASE("transforming a buffer (experimental)", "[celerity::algorithm::experimental]")
{
    distr_queue q;

    SECTION("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;
        std::vector<int> v(size, 1);
        buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});

        SECTION("tripling all elements into another buffer")
        {
            namespace h = hla::experimental;

            buffer<int, 1> buf_out(buf.get_range());

            h::transform(distr<class _346>(q), begin(buf), end(buf), begin(buf_out),
                         [](h::AnyBlock auto x) { x.configure({1}); return 3 * (*x); });

            THEN("every element is 3 in the target buffer")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<3>(r));
            }
        }
    }
}

TEST_CASE("sequencing (experimental)", "[celerity::algorithm::experimental]")
{
    distr_queue q;

    constexpr auto size = 100;
    std::vector<int> v(size, 1);
    buffer<int, 1> buf(v.data(), cl::sycl::range<1>{v.size()});

    SECTION("tripling all elements into another buffer")
    {
        namespace h = hla::experimental;

        const auto t = h::transform<class _87>([](h::AnyBlock auto x) { x.configure({1}); return 3 * (*x); });

        auto buf_out = buf | t | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<3>(r));
        }
    }

    SECTION("tripling the first element into another buffer")
    {
        namespace h = hla::experimental;

        const auto t = h::transform<class _100>([](h::All auto x) { return 3 * x[{0}]; });

        auto buf_out = buf | t | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<3>(r));
        }
    }

    SECTION("tripling the first element into another buffer")
    {
        namespace h = hla::experimental;

        const auto t = h::transform<class _115>([](h::AnySlice auto x) { x.configure(0); return 3 * *x; });

        auto buf_out = buf | t | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<3>(r));
        }
    }

    SECTION("tripling the first element into another buffer")
    {
        namespace h = hla::experimental;

        const auto t = h::zip<class _130>(std::plus<int>{});

        auto buf_out = buf | (t << buf) | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<2>(r));
        }
    }

    SECTION("tripling all elements twice")
    {
        namespace h = hla::experimental;

        const auto t = h::transform<class _146>([](h::AnyBlock auto x) { x.configure({1}); return 3 * (*x); });

        auto buf_out = buf | t | t | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<9>(r));
        }
    }

    SECTION("tripling the first element into another buffer twice")
    {
        namespace h = hla::experimental;

        const auto t = h::zip<class _130>(std::plus<int>{});

        auto buf_out = buf | (t << buf) | (t << buf) | celerity::algorithm::submit_to(q);

        THEN("every element is 3 in the target buffer")
        {
            const auto r = copy_to_host(q, buf_out);
            REQUIRE(elements_equal_to<3>(r));
        }
    }
}