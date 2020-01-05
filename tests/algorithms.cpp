#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

SCENARIO("filling a buffer", "[celerity::algorithm]")
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
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 1; }));
            }
        }

        WHEN("filling with 2s using piping syntax")
        {
            buf | fill<class fill_twos>(q, {}, 2) | q;

            THEN("the buffer contains only 2s")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 2; }));
            }
        }

        WHEN("filling with numbers 1 to 100")
        {
            generate<class iota>(q, buf, [](cl::sycl::item<1> i) { return i.get_linear_id() + 1; });

            THEN("the buffer contains numbers 1 to 100")
            {
                const auto r = copy_to_host(q, buf);
                REQUIRE(std::accumulate(begin(r), end(r), 0, std::plus<int>()) == size * (size + 1) / 2);
            }
        }
    }
}

SCENARIO("transforming a buffer", "[celerity::algorithm]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf(cl::sycl::range<1>{size});

        fill<class fill_x>(q, buf, 1);

        WHEN("tripling all elements into another buffer")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class mul3>(q, buf, buf_out, [](int x) { return 3 * x; });

            THEN("every element is 3 in the target buffer")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 3; }));
            }
        }

        WHEN("adding chunks of 3")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class add_chunk>(q, buf, buf_out, [r = buf.get_range()](chunk_i<3> x) {
                if (x.is_on_boundary(r))
                    return 2;

                return x[{-1}] + x[{}] + x[{1}];
            });

            THEN("the first and last element are 2, the rest are 3")
            {
                const auto r = copy_to_host(q, buf_out);

                REQUIRE(r.front() == 2);
                REQUIRE(r.back() == 2);
                REQUIRE(std::all_of(next(begin(r)), prev(end(r)), [](auto x) { return x == 3; }));
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

            transform<class add_ab>(q, buf_a, buf_b, buf_c, std::plus<int>{});

            THEN("every element is 5")
            {
                const auto r = copy_to_host(q, buf_c);
                REQUIRE(std::all_of(begin(r), end(r), [](auto x) { return x == 5; }));
            }
        }
    }

    GIVEN("Two two-dimensional buffers of 64x64 each representing the identity matrix")
    {
        constexpr auto rank = 64;

        std::vector<int> identity(rank * rank);
        for (size_t i = 0; i < rank; ++i)
        {
            for (size_t j = 0; j < rank; ++j)
            {
                identity[i * rank + j] = (i == j);
            }
        }

        buffer<int, 2> buf_a(identity.data(), cl::sycl::range<2>{rank, rank});
        buffer<int, 2> buf_b(identity.data(), cl::sycl::range<2>{rank, rank});

        WHEN("multiplying")
        {
            buffer<int, 2> buf_c(buf_a.get_range());

            transform<class matrix_mul>(q, buf_a, buf_b, buf_c, [](slice_i<1> a, slice_i<0> b) {
                auto sum = 0;

                for (auto k = 0; k < rank; ++k)
                {
                    const auto a_ik = a[k];
                    const auto b_kj = b[k];

                    sum += a_ik * b_kj;
                }

                return sum;
            });

            THEN("the result is the identity matrix")
            {
                const auto r = copy_to_host(q, buf_c);

                for (size_t i = 0; i < rank; ++i)
                {
                    for (size_t j = 0; j < rank; ++j)
                    {
                        const float correct_value = (i == j);

                        REQUIRE(r[j * rank + i] == correct_value);
                    }
                }
            }
        }
    }
}