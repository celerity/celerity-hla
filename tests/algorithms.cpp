#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

template <auto N>
constexpr inline auto sum_one_to_n = N *(N + 1) / 2;

SCENARIO("copying a buffer", "[celerity::algorithm]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;

        std::vector<int> src(size, 1);
        buffer<int, 1> buf(src.data(), {size});

        WHEN("copying to host")
        {
            const auto r = copy_to_host(q, buf);

            THEN("host and source buffer are equal")
            {
                REQUIRE(std::equal(begin(src), end(src), begin(r)));
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 1s")
    {
        constexpr auto rank = 10;

        std::vector<int> src(rank * rank, 1);
        buffer<int, 2> buf(src.data(), {rank, rank});

        WHEN("copying to host")
        {
            const auto r = copy_to_host(q, buf);

            THEN("host and source buffer are equal")
            {
                REQUIRE(std::equal(begin(src), end(src), begin(r)));
            }
        }
    }

    GIVEN("A three-dimensional buffer of 10x10x10 1s")
    {
        constexpr auto rank = 10;

        std::vector<int> src(rank * rank * rank, 1);
        buffer<int, 3> buf(src.data(), {rank, rank, rank});

        WHEN("copying to host")
        {
            const auto r = copy_to_host(q, buf);

            THEN("host and source buffer are equal")
            {
                REQUIRE(std::equal(begin(src), end(src), begin(r)));
            }
        }
    }
}

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

            transform<class add_ab>(q, buf_a, buf_b, buf_c, std::plus<int>{});

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

            transform<class mul4>(q, buf, buf_out, [](int x) { return 4 * x; });

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

            transform<class add_ab_10x10>(q, buf_a, buf_b, buf_c, std::plus<int>{});

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

            transform<class mul5>(q, buf, buf_out, [](int x) { return 5 * x; });

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

            // TODO: may depend on lambda capture size
            //       try to reduce size of any_accessor<> => should be the same for two-dimensional case
            //
            // ========= Program hit cudaErrorLaunchOutOfResources (error 7) due to "too many resources requested for launch" on CUDA API call to cudaLaunch.
            // =========     Saved host backtrace up to driver entry point at error
            // =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x3ac5a3]
            // =========     Host Frame:/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so.10.0 (cudaLaunch + 0x17e) [0x3801e]
            // =========     Host Frame:./build/tests/algorithms [0xaa4df]
            // =========     Host Frame:./build/tests/algorithms [0xaa3f0]
            // =========     Host Frame:./build/tests/algorithms [0xaa02d]
            // =========     Host Frame:/home/ftischler/.lib/hipSYCL/lib/libhipSYCL_cuda.so (_ZN2cl4sycl6detail15task_graph_node6submitEv + 0x2a) [0x13ffa]
            // =========     Host Frame:/home/ftischler/.lib/hipSYCL/lib/libhipSYCL_cuda.so (_ZN2cl4sycl6detail10task_graph13process_graphEv + 0xa5) [0x15095]
            // =========     Host Frame:/home/ftischler/.lib/hipSYCL/lib/libhipSYCL_cuda.so [0x15654]
            // =========     Host Frame:/home/ftischler/.lib/hipSYCL/lib/libhipSYCL_cuda.so (_ZN2cl4sycl6detail13worker_thread4workEv + 0x22b) [0x17b4b]
            // =========     Host Frame:/usr/lib/x86_64-linux-gnu/libstdc++.so.6 [0xbd6ef]
            // =========     Host Frame:/lib/x86_64-linux-gnu/libpthread.so.0 [0x76db]
            // =========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (clone + 0x3f) [0x12188f]

            transform<class add_ab_10x10x10>(q, buf_a, buf_b, buf_c, std::plus<int>{});

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

// SCENARIO("iterating a buffer on the master", "[celerity::algorithm]")
// {
//     distr_queue q;

//     GIVEN("A one-dimensional buffer of a hundred 1s")
//     {
//         constexpr auto size = 100;

//         buffer<int, 1> buf(cl::sycl::range<1>{size});

//         //fill<class fill_x_1>(q, buf, 1);

//         WHEN("checking if all are 1")
//         {
//             auto all_one = true;
//             auto checked = 0;

//             for_each(master_blocking(q), buf, [&](cl::sycl::item<1> it, int x) {
//                 all_one = all_one && x == 1;
//                 checked++;
//             });

//             THEN("the outcome is true")
//             {
//                 REQUIRE(all_one);
//                 REQUIRE(checked == 100);
//             }
//         }
//     }

    // GIVEN("A three-dimensional buffer of 10x10x10 1s")
    // {
    //     constexpr auto size = 10;

    //     buffer<int, 3> buf({size, size, size});

    //     fill<class fill_x_2>(q, buf, 1);

    //     WHEN("checking if all are 1")
    //     {
    //         auto all_one = true;
    //         auto checked = 0;

    //         for_each(master_blocking(q), buf, [&](cl::sycl::item<3> it, int x) {
    //             all_one = all_one && x == 1;
    //             checked++;
    //         });

    //         THEN("the outcome is true")
    //         {
    //             REQUIRE(all_one);
    //             REQUIRE(checked == 1000);
    //         }
    //     }
    // }

    // GIVEN("A one-dimensional buffer of 100 1s")
    // {
    //     constexpr auto size = 100;

    //     buffer<int, 1> buf(cl::sycl::range<1>{size});

    //     fill<class fill_x_3>(q, buf, 1);

    //     WHEN("checking if all are 1 using master all<> access")
    //     {
    //         auto all_one = false;
    //         auto checked = 0;

    //         master_task(
    //             master(q), buf, [&](algorithm::all<int, 1> b) {
    //                 checked = 1000;
    //                 all_one = std::all_of(begin(b), end(b), [](int x) { return x == 1; });
    //             });

    //         q.slow_full_sync();

    //         THEN("the outcome is true")
    //         {
    //             REQUIRE(checked == 1000);
    //             REQUIRE(all_one);
    //         }
    //     }
    // }

    // GIVEN("Two one-dimensional buffer of 100 1s")
    // {
    //     constexpr auto size = 100;

    //     buffer<int, 1> buf_a(cl::sycl::range<1>{size});
    //     buffer<int, 1> buf_b(cl::sycl::range<1>{size});

    //     fill<class fill_x_4>(q, buf_a, 1);
    //     fill<class fill_x_5>(q, buf_b, 2);

    //     WHEN("checking if all are 1 using master all<> access")
    //     {
    //         auto all_one = false;
    //         auto checked = 0;

    //         master_task(master(q), pack(buf_a, buf_b), [&](algorithm::all<int, 1> a, algorithm::all<int, 1> b) {
    //             checked = 1000;
    //             all_one = std::all_of(begin(a), end(a), [](int x) { return x == 1; });
    //             all_one = all_one && std::all_of(begin(b), end(b), [](int x) { return x == 2; });
    //         });

    //         q.slow_full_sync();

    //         THEN("the outcome is true")
    //         {
    //             REQUIRE(checked == 1000);
    //             REQUIRE(all_one);
    //         }
    //     }
    // }
//}