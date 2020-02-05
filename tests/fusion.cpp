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

        WHEN("chaining calls targeting same input/output buffers")
        {
            //celerity::buffer<int, 1> buf_out(buf_in.get_range());
 
            auto t1 = transform<class add>(q, {}, {}, {}, add_5);
            auto t2 = transform<class mul>(q, {}, {}, {}, mul_3);

            // auto t1 = transform<class add>(q, buf_in, buf_out, add_5);
            // auto t2 = transform<class mul>(q, buf_in, buf_out, mul_3);

            // t1.disable();
            // t2.disable();

            // auto lhs = t1.get_sequence();
            // auto rhs = t2.get_sequence();

            // auto seq = lhs.get_task().get_sequence() | rhs.get_task().get_sequence();

            // auto f = [=](handler& cgh)
            // {
            //     auto kernels = seq(cgh);

            //     return [=](cl::sycl::item<1> item)
            //     {
            //         return sequence(kernels);
            //     };
            // };

            // auto t = task<named_distributed_execution_policy<class fused_kernels>>(f);

            // t(q, begin(buf_in), end(buf_in));

            //auto seq = lhs | t2;
 
            auto seq = buf_in | t1 | t2; 
            auto buf_out = seq | submit_to(q);
 
            //static_assert(algorithm::detail::is_packaged_task_v<seq_t>);

            //seq(q);

            THEN("kernels get fused and the result is 18")
            {
                using seq_t = decltype(seq);
                static_assert(algorithm::detail::is_packaged_task_v<seq_t>);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(next(begin(r)), prev(end(r))));
            }
        }  
    }
}