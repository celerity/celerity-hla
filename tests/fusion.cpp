#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include "../src/sequencing.h"
#include "../src/actions.h"
#include "../src/fusion_helper.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;
using namespace celerity::algorithm::traits;
using namespace celerity::algorithm::util;
using celerity::algorithm::chunk;

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
            auto t1 = transform<class add>(add_5);
            auto t2 = transform<class mul>(mul_3);

            auto seq = buf_in | t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(begin(r), end(r)));
            }
        }
    }

    GIVEN("Two simple transform kernels and a buffer of hundred 1s also taking the current item")
    {
        auto add_5 = [](cl::sycl::item<1>, int x) { return x + 5; };
        auto mul_3 = [](cl::sycl::item<1>, int x) { return x * 3; };

        constexpr auto size = 100;
        std::vector<int> src(size, 1);
        celerity::buffer<int, 1> buf_in(src.data(), {size});

        WHEN("chaining calls")
        {
            auto t1 = transform<class add_w_item>(add_5);
            auto t2 = transform<class mul_w_item>(mul_3);

            auto seq = buf_in | t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(begin(r), end(r)));
            }
        }
    }

    GIVEN("A generate and a transform kernel")
    {
        constexpr auto size = 100;

        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto mul_3 = [](int x) { return x * 3; };

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class gen_item_id>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class mul_3_f>(mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t1 = fill_n<class fill_1>(cl::sycl::range<1>{size}, init);
            auto t2 = transform<class mul_3_f_>(mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t1 = fill_n<class fill_1>(cl::sycl::range<1>{size}, init);
            auto t2 = transform<class add_w_item>(add_5);
            auto t3 = transform<class mul_w_item>(mul_3);

            auto seq = t1 | t2 | t3;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 3);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t1 = generate_n<class generate_item_id>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class n_add_7>(add_7);
            auto t3 = transform<class n_mul_4>(mul_4);
            auto t4 = transform<class n_div_2>(div_2);

            auto seq = t1 | t2 | t3 | t4;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is correct")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 4);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t1 = generate_n<class generate_item_id>(cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class zip_add_t>(zip_add);

            auto seq = t1 | (t2 << buf_in);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t3 = transform<class zip_add_t_1>(zip_add);

            auto seq = t1 | (t3 << t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
            auto t3 = transform<class zip_add_t_2>(zip_add);
            auto t4 = transform<class mul_2_t_1>(mul_2);

            auto seq = t1 | (t3 << t2) | t4;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 3);
                static_assert(size_v<fused_sequence_type> == 1); // TODO

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
            auto t3 = transform<class zip_add_t_3>(zip_add);
            auto t4 = transform<class mul_2_t_2>(mul_2);
            auto t5 = transform<class add_3_t_2>(add_3);

            auto seq = t1 | (t3 << t2) | t4 | t5;
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 4);
                static_assert(size_v<fused_sequence_type> == 1); // TODO

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
        fill<class _1>(q, begin(buf), end(buf), 1);

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_3>(cl::sycl::range<1>{size}, gen_i);
            auto t3 = transform<class zip_add_t_3>(zip_add);
            auto seq = t1 | (t3 << buf);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);
                static_assert(computation_type_of_v<last_element_t<fused_sequence_type>, computation_type::generate>);

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
        fill<class _2>(q, begin(buf), end(buf), 1);

        WHEN("chaining calls")
        {
            auto t1 = generate_n<class generate_item_id_3>(cl::sycl::range<1>{size}, gen_i);
            auto t3 = transform<class zip_add_t_4>(zip_add);
            auto seq = buf | (t3 << t1);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);
                //static_assert(is_t_joint_v<fused_sequence_type>);
                static_assert(!has_transient_input_v<last_element_t<fused_sequence_type>>);
                static_assert(has_transient_second_input_v<last_element_t<fused_sequence_type>>);

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
        fill<class _3>(q, begin(buf_a), end(buf_a), 1);
        fill<class _4>(q, begin(buf_b), end(buf_b), 2);

        WHEN("chaining calls")
        {
            auto t3 = transform<class zip_add_t_5>(zip_add);
            auto seq = buf_a | (t3 << buf_b);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);
                static_assert(computation_type_of_v<last_element_t<fused_sequence_type>, computation_type::zip>);
                static_assert(!has_transient_input_v<last_element_t<fused_sequence_type>>);
                static_assert(!has_transient_second_input_v<last_element_t<fused_sequence_type>>);

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
        fill<class _5>(q, begin(buf_a), end(buf_a), 1);
        fill<class _6>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_5>(mul_chunk);
            auto t3 = transform<class zip_add_t_6>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 20")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
        fill<class _7>(q, begin(buf_a), end(buf_a), 1);
        fill<class _8>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_7>(mul_chunk);
            auto t3 = transform<class zip_add_t_8>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 80")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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
        fill<class _9>(q, begin(buf_a), end(buf_a), 1);
        fill<class _10>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_9>(mul_chunk);
            auto t3 = transform<class zip_add_t_10>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("kernels are fused and the result is 80")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](const chunk<int, 1> &x, int y) { return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _11>(q, begin(buf_a), end(buf_a), 1);
        fill<class _12>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_11>(mul_chunk);
            auto t3 = transform<class zip_add_t_12>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 80")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 2);

                static_assert(!is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](int x, const chunk<int, 1> &y) { return x + *y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _13>(q, begin(buf_a), end(buf_a), 1);
        fill<class _14>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_13>(mul_chunk);
            auto t3 = transform<class zip_add_t_14>(zip_add);
            auto seq = buf_a | t2 | t3 << (buf_b | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 20")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](const chunk<int, 1> &x, int y) { return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _15>(q, begin(buf_a), end(buf_a), 1);
        fill<class _16>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_15>(mul_chunk);
            auto t3 = transform<class zip_add_t_16>(zip_add);
            auto seq = buf_a | t3 << (buf_b | t2 | t2);
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 76")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(!is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add = [](const chunk<int, 1> &x, int y) { return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _17>(q, begin(buf_a), end(buf_a), 1);
        fill<class _18>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_17>(mul_chunk);
            auto t3 = transform<class zip_add_t_18>(zip_add);
            auto secondary_sequence = buf_a | t3 << (buf_b | t2 | t2);
            auto seq = buf_a | t3 << secondary_sequence;
            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 77")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(!is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, const chunk<int, 1> &y) { return x + *y; };
        auto zip_add = [](const chunk<int, 1> &x, int y) { return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _19>(q, begin(buf_a), end(buf_a), 1);
        fill<class _20>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_19>(mul_chunk);
            auto t3 = transform<class zip_add_t_20>(zip_add);
            auto t4 = transform<class zip_add_sec_1>(zip_add_sec);
            auto secondary_sequence = buf_a | t4 << (buf_b | t2);
            auto seq = buf_a | t3 << secondary_sequence;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 17")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](const chunk<int, 1> &x, const chunk<int, 1> &y) { return *x + *y; };
        auto zip_add = [](const chunk<int, 1> &x, int y) { return *x + y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _21>(q, begin(buf_a), end(buf_a), 1);
        fill<class _22>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_21>(mul_chunk);
            auto t3 = transform<class zip_add_t_22>(zip_add);
            auto t4 = transform<class zip_add_sec_2>(zip_add_sec);
            auto secondary_sequence = buf_a | t2 | t4 << (buf_b | t2);
            auto seq = buf_a | t3 << secondary_sequence;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 21")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 1);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_sequence_type>>);

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

        using celerity::algorithm::chunk;
        using celerity::algorithm::detail::access_type;

        auto mul_chunk = [](int c) { return c * 5; };
        auto zip_add_sec = [](int x, const chunk<int, 1> &y) { return x + *y; };

        buffer<int, 1> buf_a{{size}};
        buffer<int, 1> buf_b{{size}};
        fill<class _23>(q, begin(buf_a), end(buf_a), 1);
        fill<class _24>(q, begin(buf_b), end(buf_b), 3);

        WHEN("chaining calls")
        {
            auto t2 = transform<class mul_chunk_23>(mul_chunk);
            auto t4 = transform<class zip_add_sec_3>(zip_add_sec);
            auto seq = buf_a | t4 << (buf_b | t2) | t2;

            auto buf_out = seq | submit_to(q);

            THEN("secondary sequence is fused and the result is 80")
            {
                using terminated_sequence_type = decltype(terminate(seq));
                using fused_sequence_type = decltype(fuse(std::declval<terminated_sequence_type>()));
                using algorithm::detail::computation_type;

                static_assert(size_v<terminated_sequence_type> == 2);
                static_assert(size_v<fused_sequence_type> == 1);

                static_assert(is_t_joint_v<last_element_t<fused_sequence_type>>);

                const auto r = copy_to_host(q, buf_out);

                for (auto i = 5; i < size; ++i)
                {
                    REQUIRE(r[i] == 80);
                }
            }
        }
    }

}