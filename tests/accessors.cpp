#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

SCENARIO("accessing a slice", "[accessors::slice]")
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

SCENARIO("accessing a chunk", "[accessors::chunk]")
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

    GIVEN("A two-dimensional buffer of 10x10 1s")
{
        constexpr auto rank = 10;

        buffer<int, 2> buf({rank, rank});
        fill<class fill_10x10>(q, buf, 1);

        WHEN("")

        WHEN("summing chunks of 3x3")
        {
            constexpr auto chunk_size = 3;

            buffer<int, 2> buf_out(buf.get_range());

            transform<class sum_chunk_3x3>(q, buf, buf_out, [r = buf.get_range()](chunk_i<chunk_size, chunk_size> c) {
                if (c.is_on_boundary(r))
                    return 0;

                auto sum = 0;

                for (auto y = -(chunk_size / 2); y <= chunk_size / 2; ++y)
                {
                    for (auto x = -(chunk_size / 2); x <= chunk_size / 2; ++x)
                    {
                        sum += c[{y, x}];
                    }
                }

                return sum;
            });

            THEN("border elements are 0, the others are 9")
            {
                const auto r = copy_to_host(q, buf_out);

                for (size_t y = 0; y < rank; ++y)
                {
                    for (size_t x = 0; x < rank; ++x)
                    {
                        const auto correct_value = (x == 0 || x == rank - 1 ||
                                                    y == 0 || y == rank - 1)
                                                       ? 0
                                                       : 9;

                        REQUIRE(r[y * rank + x] == correct_value);
                    }
                }
            }
        }
    }
}

SCENARIO("accessing the entire buffer", "[accessors::all]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of one 2 and ninety-nine 1s")
    {
        constexpr size_t size = 100;

        std::vector<int> src(size, 1);
        src.front() = 2;

        buffer<int, 1> buf(src.data(), cl::sycl::range<1>{size});

        WHEN("assigning all elements the value of the first")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class set_first>(q, buf, buf_out, [](all_i x) {
                return x[{0}];
            });

            THEN("each element is equal to 2")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<2>(r));
            }
        }

        /*constexpr auto any_acc_size = sizeof(celerity::detail::any_accessor<int>);
        constexpr auto slice_size = sizeof(slice_i<0>);
        constexpr auto chunk_size = sizeof(chunk_i<3>);
        constexpr auto all_size = sizeof(all_i);
        constexpr auto acc_size = sizeof(celerity::detail::device_accessor<double, 3, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>);
*/
        WHEN("assigning the sum of all elements to all elements")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class set_sum>(q, buf, buf_out, [r = buf.get_range()](all_i x) {
                auto sum = 0;

                for (size_t i = 0; i < size; ++i)
                    sum += x[{i}];

                return sum;
            });

            THEN("each element is equal to 101")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<101>(r));
            }
        }
    }
}