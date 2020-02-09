#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

#pragma clang diagnostic warning "-Wall"

#include "../src/actions.h"

//#include "../src/fusion.h"

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
            auto t1 = transform<class add>(q, {}, {}, {}, add_5);
            auto t2 = transform<class mul>(q, {}, {}, {}, mul_3);

            auto seq = buf_in | t1 | t2;
            auto buf_out = seq | submit_to(q);

            // short
            //
            // auto buf_out = transform<class add>(q, {}, {}, {}, add_5) |
            //                transform<class mul>(q, {}, {}, {}, mul_3) |
            //                submit_to(q)

            THEN("kernels are fused and the result is 18")
            {
                /*using seq_t = decltype(seq);
                static_assert(algorithm::detail::is_packaged_task_v<seq_t> ||
                              (algorithm::detail::is_packaged_task_sequence_v<seq_t> && size_v<seq_t> == 1));
                */

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
            auto t1 = transform<class add_w_item>(q, {}, {}, {}, add_5);
            auto t2 = transform<class mul_w_item>(q, {}, {}, {}, mul_3);

            auto seq = buf_in | t1 | t2;
            auto buf_out = seq | submit_to(q);

            // short
            //
            // auto buf_out = transform<class add>(q, {}, {}, {}, add_5) |
            //                transform<class mul>(q, {}, {}, {}, mul_3) |
            //                submit_to(q)

            THEN("kernels are fused and the result is 18")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(begin(r), end(r)));
            }
        }
    }

    GIVEN("A generate and a transform kernel")
    {
        auto gen_i = [](cl::sycl::item<1> i) { return static_cast<int>(i.get_linear_id()); };
        auto mul_3 = [](int x) { return x * 3; };

        constexpr auto size = 100;

        WHEN("chaining calls")
        {
            auto t1 = generate<class gen_item_id>(q, cl::sycl::range<1>{100}, gen_i);
            auto t2 = transform<class mul_3_f>(q, {}, {}, {}, mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            static_assert(is_transiently_linkable_source_v<decltype(t1)>);
            static_assert(is_source_v<decltype(t1)>);
            static_assert(is_linkable_source_v<decltype(t1)>);
            static_assert(algorithm::detail::is_partially_packaged_task_v<decltype(t1)>);
            static_assert(algorithm::detail::stage_requirement_v<decltype(t1)> == stage_requirement::output);

            static_assert(is_transiently_linkable_sink_v<decltype(t2)>);

            //auto buf_out = sequence(t1) | submit_to(q);

            // short
            //
            // auto buf_out = transform<class add>(q, {}, {}, {}, add_5) |
            //                transform<class mul>(q, {}, {}, {}, mul_3) |
            //                submit_to(q)

            /*static_assert(size_v<decltype(remove_last_element(seq) | (get_last_element(seq) | q))> == 2);

            using completed_sequence = decltype(get_last_element(get_last_element(seq) | q));

            static_assert(algorithm::detail::is_packaged_task_v<completed_sequence>);
            static_assert(std::is_same_v<typename completed_sequence::output_iterator_type, buffer_iterator<int, 1>>);
            static_assert(std::is_same_v<decltype(std::declval<completed_sequence>() | q), buffer<int, 1>>);
            
            using fused_seq = decltype(fuse(remove_last_element(seq) | get_last_element(get_last_element(seq) | q)));
            static_assert(std::is_same_v<decltype(std::declval<fused_seq>() | q), buffer<int, 1>>);*/

            THEN("kernels are fused and the result is 3")
            {
                //using seq_t = decltype(fuse(seq));
                //static_assert(algorithm::detail::is_packaged_task_v<seq_t> || (algorithm::detail::is_packaged_task_sequence_v<seq_t> && size_v<seq_t> == 1));

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
        constexpr auto init = 1;
        auto mul_3 = [](int x) { return x * 3; };

        constexpr auto size = 100;

        WHEN("chaining calls")
        {
            auto t1 = fill<class fill_1>(q, cl::sycl::range<1>{100}, init);
            auto t2 = transform<class mul_3_f_>(q, {}, {}, {}, mul_3);

            auto seq = t1 | t2;
            auto buf_out = seq | submit_to(q);

            static_assert(is_transiently_linkable_source_v<decltype(t1)>);
            static_assert(is_source_v<decltype(t1)>);
            static_assert(is_linkable_source_v<decltype(t1)>);
            static_assert(algorithm::detail::is_partially_packaged_task_v<decltype(t1)>);
            static_assert(algorithm::detail::stage_requirement_v<decltype(t1)> == stage_requirement::output);

            static_assert(is_transiently_linkable_sink_v<decltype(t2)>);

            //auto buf_out = sequence(t1) | submit_to(q);

            // short
            //
            // auto buf_out = transform<class add>(q, {}, {}, {}, add_5) |
            //                transform<class mul>(q, {}, {}, {}, mul_3) |
            //                submit_to(q)

            /*static_assert(size_v<decltype(remove_last_element(seq) | (get_last_element(seq) | q))> == 2);

            using completed_sequence = decltype(get_last_element(get_last_element(seq) | q));

            static_assert(algorithm::detail::is_packaged_task_v<completed_sequence>);
            static_assert(std::is_same_v<typename completed_sequence::output_iterator_type, buffer_iterator<int, 1>>);
            static_assert(std::is_same_v<decltype(std::declval<completed_sequence>() | q), buffer<int, 1>>);
            
            using fused_seq = decltype(fuse(remove_last_element(seq) | get_last_element(get_last_element(seq) | q)));
            static_assert(std::is_same_v<decltype(std::declval<fused_seq>() | q), buffer<int, 1>>);*/

            THEN("kernels are fused and the result is 3")
            {
                //using seq_t = decltype(fuse(seq));
                //static_assert(algorithm::detail::is_packaged_task_v<seq_t> || (algorithm::detail::is_packaged_task_sequence_v<seq_t> && size_v<seq_t> == 1));

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<3 * init>(begin(r), end(r)));
            }
        }
    }
}