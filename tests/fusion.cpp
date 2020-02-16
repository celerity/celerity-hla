#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

#pragma clang diagnostic warning "-Wall"

#include "../src/actions.h"

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

            auto buf_out = buf_in | t1 | t2 | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(buf_in | t1 | t2));
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
            auto t1 = transform<class add_w_item>(q, {}, {}, {}, add_5);
            auto t2 = transform<class mul_w_item>(q, {}, {}, {}, mul_3);

            auto buf_out = buf_in | t1 | t2 | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(buf_in | t1 | t2));
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
            auto t1 = generate<class gen_item_id>(q, cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class mul_3_f>(q, {}, {}, {}, mul_3);

            auto buf_out = t1 | t2 | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using terminated_sequence_type = decltype(terminate(t1 | t2));
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
            auto t1 = fill<class fill_1>(q, cl::sycl::range<1>{size}, init);
            auto t2 = transform<class mul_3_f_>(q, {}, {}, {}, mul_3);

            auto buf_out = t1 | t2 | submit_to(q);

            THEN("kernels are fused and the result is 3")
            {
                using terminated_sequence_type = decltype(terminate(t1 | t2));
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
            auto t1 = fill<class fill_1>(q, cl::sycl::range<1>{size}, init);
            auto t2 = transform<class add_w_item>(q, {}, {}, {}, add_5);
            auto t3 = transform<class mul_w_item>(q, {}, {}, {}, mul_3);

            auto buf_out = t1 | t2 | t3 | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(t1 | t2 | t3));
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
            auto t1 = generate<class generate_item_id>(q, cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class n_add_7>(q, {}, {}, {}, add_7);
            auto t3 = transform<class n_mul_4>(q, {}, {}, {}, mul_4);
            auto t4 = transform<class n_div_2>(q, {}, {}, {}, div_2);

            auto buf_out = t1 | t2 | t3 | t4 | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(t1 | t2 | t3 | t4));
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
            auto t1 = generate<class generate_item_id>(q, cl::sycl::range<1>{size}, gen_i);
            auto t2 = transform<class zip_add_t>(q, zip_add);

            auto buf_out = t1 | (t2 << buf_in) | submit_to(q);

            THEN("kernels are fused and the result is 18")
            {
                using terminated_sequence_type = decltype(terminate(t1 | (t2 << buf_in)));
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
            auto t1 = generate<class generate_item_id_1>(q, cl::sycl::range<1>{size}, gen_i);
            auto t2 = fill<class fill_3>(q, cl::sycl::range<1>(size), 3);
            auto t3 = transform<class zip_add_t_1>(q, zip_add);

            std::cout << to_string(terminate(t1 | (t3 << t2))) << std::endl;

            std::cout << "Fused:\n\n";
            std::cout << to_string(fuse(terminate(t1 | (t3 << t2)))) << std::endl;

            auto buf_out = t1 | (t3 << t2) | submit_to(q);

            THEN("kernels are fused and the result is (i + 3)") 
            { 
                using terminated_sequence_type = decltype(terminate(t1 | (t3 << t2)));
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
}