#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include "../include/sequencing.h"
#include "../include/actions.h"
#include "../include/fusion_helper.h"

#include "../include/experimental/algorithms/transform.h"
#include "../include/experimental/algorithms/zip.h"
#include "../include/experimental/algorithms/generate.h"
#include "../include/experimental/algorithms/fill.h"

#include <numeric>

using namespace celerity;
using namespace celerity::hla::experimental;
using namespace celerity::hla::traits;
using namespace celerity::hla::util;

using hla::chunk;
using hla::submit_to;
using hla::operator|;
using hla::operator<<;

using hla::fused_t;
using hla::linked_t;
using hla::traits::computation_type_of_v;
using hla::traits::has_transient_input_v;
using hla::traits::has_transient_second_input_v;
using hla::traits::is_t_joint_v;
using hla::traits::size_v;

using hla::experimental::fill;

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

        WHEN("chaining calls")
        {
            auto t1 = transform<class _44>(add_5);
            auto t2 = transform<class _45>(mul_3);

            auto seq = buf_in | t1 | t2;

            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(begin(r), end(r)));
            }
        }
    }

    // GIVEN("Two simple transform kernels and a buffer of hundred 1s also taking the current item")
    // {
    //     auto add_5 = [](cl::sycl::item<1>, int x) { return x + 5; };
    //     auto mul_3 = [](cl::sycl::item<1>, int x) { return x * 3; };

    //     constexpr auto size = 100;
    //     std::vector<int> src(size, 1);
    //     celerity::buffer<int, 1> buf_in(src.data(), {size});

    //     WHEN("chaining calls")
    //     {
    //         auto t1 = transform<class _75>(add_5);
    //         auto t2 = transform<class _76>(mul_3);

    //         auto seq = buf_in | t1 | t2;
    //         auto buf_out = seq | submit_to(q);

    //         THEN("kernels are fused and the result is 18")
    //         {
    //             using seq_type = decltype(seq);

    //             static_assert(size_v<linked_t<seq_type>> == 2);
    //             static_assert(size_v<fused_t<seq_type>> == 1);

    //             const auto r = copy_to_host(q, buf_out);
    //             REQUIRE(elements_equal_to<18>(begin(r), end(r)));
    //         }
    //     }
    // }

    GIVEN("A generate and a transform kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto mul_3 = [](int x) { return x * 3; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class _103>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class _104>(mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 0; i < size; ++i)
                {
                    REQUIRE(r[i] == i * 3);
                }
            }
        }
    }

    GIVEN("A fill value and a transform kernel")
    {
        constexpr auto size = 100;
        constexpr auto init = 1;

        auto mul_3 = [](int x) { return x * 3; };

        WHEN("chaining calls")
        {
            auto t1 = fill_n<class _135>(cl::sycl::range<1>{size}, init);
            auto t2 = transform<class _136>(mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<3 * init>(begin(r), end(r)));
            }
        }
    }

    GIVEN("A fill value and two simple transform kernels")
    {
        constexpr auto size = 100;
        constexpr auto init = 1;

        auto add_5 = [](int x) { return x + 5; };
        auto mul_3 = [](int x) { return x * 3; };

        WHEN("chaining calls")
        {
            auto t1 = fill_n<class _164>(cl::sycl::range<1>{size}, init);
            auto t2 = transform<class _165>(add_5);
            auto t3 = transform<class _166>(mul_3);

            auto seq = t1 | t2 | t3;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 3);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(begin(r), end(r)));
            }
        }
    }

    GIVEN("A generate and three transform kernels")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto add_7 = [](int x) { return x + 7; };
        auto mul_4 = [](int x) { return x * 4; };
        auto div_2 = [](int x) { return x / 2; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class _195>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class _196>(add_7);
            auto t3 = transform<class _197>(mul_4);
            auto t4 = transform<class _198>(div_2);

            auto seq = t1 | t2 | t3 | t4;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is correct")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 4);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 0; i < size; ++i)
                {
                    REQUIRE(r[i] == div_2(mul_4(add_7(i))));
                }
            }
        }
    }

    GIVEN("A generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };

        std::vector<int> src(size, 1);
        celerity::buffer<int, 1> buf_in(src.data(), {size});

        auto zip_add = [](int x, int y) { return x + y; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class _233>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = zip<class _243>(zip_add);

            auto seq = t1 | (t2 << buf_in);

            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 0; i < size; ++i)
                {
                    REQUIRE(r[i] == zip_add(1, i));
                }
            }
        }
    }

    GIVEN("Two generate kernels and a zip kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto zip_add = [](int x, int y) { return x + y; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_1>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = fill_n<class fill_3>(cl::sycl::range<1>(size), 3);
            auto t3 = zip<class _268>(zip_add);

            auto seq = t1 | (t3 << t2);
            static_assert(!celerity::hla::traits::is_sequence_v<celerity::hla::traits::first_element_t<decltype(seq)>>);
            static_assert(!celerity::hla::traits::is_sequence_v<celerity::hla::traits::last_element_t<decltype(seq)>>);

            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == zip_add(3, i));
                }
            }
        }
    }

    GIVEN("Two generate kernels a zip kernel and a transform kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto zip_add = [](int x, int y) { return x + y; };
        auto mul_2 = [](int x) { return 2 * x; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_2>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = fill_n<class fill_4>(cl::sycl::range<1>(size), 3);
            auto t3 = zip<class zip_add_t_2>(zip_add);
            auto t4 = transform<class mul_2_t_1>(mul_2);

            auto seq = t1 | (t3 << t2) | t4;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 3);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == mul_2(zip_add(3, i)));
                }
            }
        }
    }

    GIVEN("Two generate kernels a zip kernel and two transform kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto zip_add = [](int x, int y) { return x + y; };
        auto mul_2 = [](int x) { return 2 * x; };
        auto add_3 = [](int x) { return x + 3; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_3>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = fill_n<class fill_5>(cl::sycl::range<1>(size), 3);
            auto t3 = zip<class _343>(zip_add);
            auto t4 = transform<class mul_2_t_2>(mul_2);
            auto t5 = transform<class add_3_t_2>(add_3);

            auto seq = t1 | (t3 << t2) | t4 | t5;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 4);
                static_assert(size_v<fused_t<seq_type>> == 1); // TODO

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == add_3(mul_2(zip_add(3, i))));
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf{{size}};
        hla::experimental::fill<class _1>(q, begin(buf), end(buf), 1);

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_3>(cl::sycl::range<1>{size}, gen_i);
            auto t3 = zip<class _380>(zip_add);
            auto seq = t1 | (t3 << buf);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using seq_type = decltype(seq);

                static_assert(hla::traits::size_v<linked_t<seq_type>> == 2);
                static_assert(hla::traits::size_v<fused_t<seq_type>> == 1);
                static_assert(hla::traits::computation_type_of_v<last_element_t<fused_t<seq_type>>,
                                                                 celerity::hla::detail::computation_type::generate>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == zip_add(1, i));
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf{{size}};
        hla::experimental::fill<class _2>(q, begin(buf), end(buf), 1);

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_3>(cl::sycl::range<1>{size}, gen_i);
            auto t3 = zip<class _416>(zip_add);
            auto seq = buf | (t3 << t1);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);
                static_assert(!has_transient_input_v<last_element_t<fused_t<seq_type>>>);
                //TODO: static_assert(has_transient_second_input_v<last_element_t<fused_sequence_type>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == zip_add(1, i));
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _3>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _4>(q, begin(buf_b), end(buf_b), 2);

        WHEN("chaining calls")
        {
            auto t3 = zip<class _456>(zip_add);
            auto seq = buf_a | (t3 << buf_b);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using hla::detail::computation_type;

                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);
                static_assert(computation_type_of_v<last_element_t<fused_t<seq_type>>, computation_type::zip>);
                static_assert(!has_transient_input_v<last_element_t<fused_t<seq_type>>>);
                static_assert(!has_transient_second_input_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 3);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _5>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _6>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_5>(mul_chunk);
            auto t3 = zip<class _497>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 20")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 20);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _7>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _8>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_7>(mul_chunk);
            auto t3 = zip<class _533>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 80")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 80);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _9>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _10>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_9>(mul_chunk);
            auto t3 = zip<class _569>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 80")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 80);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _11>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _12>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_11>(mul_chunk);
            auto t3 = zip<class _610>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 80")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 2);

                static_assert(!is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 80);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](int x, Block auto y) { y.configure(1); return x + *y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _13>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _14>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_13>(mul_chunk);
            auto t3 = zip<class zip_add_t_14>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 20")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 20);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _15>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _16>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_15>(mul_chunk);
            auto t3 = zip<class _693>(zip_add);
            auto seq = buf_a | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 76")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(!is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 76);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _17>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _18>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_17>(mul_chunk);
            auto t3 = zip<class _734>(zip_add);
            auto secondary_sequence = buf_a | t3 << (buf_b | t2 | t2);
            auto seq = buf_a | t3 << secondary_sequence;
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 77")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(!is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 77);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _19>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _20>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_19>(mul_chunk);
            auto t3 = zip<class _777>(zip_add);
            auto t4 = zip<class _778>(zip_add_sec);
            auto secondary_sequence = buf_a | t4 << (buf_b | t2);
            auto seq = buf_a | t3 << secondary_sequence;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 17")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 17);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](Block auto x, Block auto y) { x.configure(1); y.configure(1); return *x + *y; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _21>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _22>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_21>(mul_chunk);
            auto t3 = zip<class _882>(zip_add);
            auto t4 = zip<class _823>(zip_add_sec);
            auto secondary_sequence = buf_a | t2 | t4 << (buf_b | t2);
            auto seq = buf_a | t3 << secondary_sequence;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 1);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 21);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _860>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _861>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_23>(mul_chunk);
            auto t4 = zip<class _866>(zip_add_sec);
            auto seq = buf_a | t4 << (buf_b | t2) | t2;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 80")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 80);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _25>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _26>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_25>(mul_chunk);
            auto t4 = zip<class _908>(zip_add_sec);
            auto seq = buf_a | t4 << (buf_b | t2) | t4 << (buf_a | t2);

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 21);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _27>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _28>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_27>(mul_chunk);
            auto t3 = zip<class _951>(zip_add);
            auto t4 = zip<class _952>(zip_add_sec);
            auto seq = buf_a | t3 << (buf_b | t2) | t4 << (buf_a | t2);

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 21);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };
        auto zip_add = [](int x, int y) { return x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _29>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _30>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_29>(mul_chunk);
            auto t3 = zip<class _995>(zip_add);
            auto t4 = zip<class _996>(zip_add_sec);
            auto seq = buf_a | t4 << (buf_b | t2) | t3 << (buf_a | t2);

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 21);
                }
            }
        }
    }

    GIVEN("One generate kernel, a buffer and a zip kernel")
    {
        constexpr auto size = 100;

        using celerity::hla::chunk;
        using celerity::hla::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, Block auto y) { y.configure(1); return x + *y; };
        auto zip_add = [](Block auto x, int y) { x.configure(1); return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        hla::experimental::fill<class _31>(q, begin(buf_a), end(buf_a), 1);
        hla::experimental::fill<class _32>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_31>(mul_chunk);
            auto t3 = zip<class _1039>(zip_add);
            auto t4 = zip<class _1040>(zip_add_sec);
            auto seq = buf_a | t4 << (buf_b | t2) | t3 << (buf_a | t2);

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using seq_type = decltype(seq);

                static_assert(size_v<linked_t<seq_type>> == 2);
                static_assert(size_v<fused_t<seq_type>> == 2);

                static_assert(is_t_joint_v<first_element_t<fused_t<seq_type>>>);
                static_assert(!is_t_joint_v<last_element_t<fused_t<seq_type>>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 21);
                }
            }
        }
    }
}