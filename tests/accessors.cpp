#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

SCENARIO("accessing a slice", "[celerity::algorithm::accessors]")
{
    distr_queue q;

    GIVEN("Two two-dimensional buffers of 64x64 both representing the identity matrix")
    {
        constexpr auto rank = 16;

        std::vector<int> identity(rank * rank);
        for (size_t i = 0; i < rank; ++i)
        {
            for (size_t j = 0; j < rank; ++j)
            {
                identity[i * rank + j] = (i == j);
            }
        }

        buffer<int, 2> buf_a(identity.data(), {rank, rank});
        buffer<int, 2> buf_b(identity.data(), {rank, rank});

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

SCENARIO("accessing a chunk", "[celerity::algorithm::accessors]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf(cl::sycl::range<1>{size});
        fill<class fill_x>(q, buf, 1);

        WHEN("adding chunks of 3")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class add_chunk>(q, buf, buf_out, [r = buf.get_range()](chunk_i<3> x) {
                if (x.is_on_boundary(r))
                    return 2;

                return x[{-1}] + *x + x[{1}];
            });

            THEN("the first and last element are 2, the others are 3")
            {
                const auto r = copy_to_host(q, buf_out);

                REQUIRE(r.front() == 2);
                REQUIRE(r.back() == 2);
                REQUIRE(elements_equal_to<3>(next(begin(r)), prev(end(r))));
            }
        }
    }
}