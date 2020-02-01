#define CATCH_CONFIG_MAIN

#include "utils.h"

#include <numeric>

using namespace celerity;
using namespace celerity::algorithm;

#include "../src/actions.h"

SCENARIO("Fusing two tasks", "[fusion::simple]")
{
    celerity::distr_queue q{};

    GIVEN("Two simple transform kernels and a buffer of hundred 1s")
    {
        auto add_5 = [](int x) { return x + 5; };
        auto mul_3 = [](int x) { return x * 3; };

        constexpr auto size = 100;
        std::vector<int> src(1, size);
        celerity::buffer<int, 1> buf_in(src.data(), {size});

        WHEN("chaining calls targeting same input/output buffers")
        {
            celerity::buffer<int, 1> buf_out(buf_in.get_range());

            auto t = transform<class add>(q, begin(buf_in), end(buf_in), begin(buf_out), add_5);

            using decorator_t = decltype(t.get_sequence());

            static_assert(is_simple_transform_decorator_v<decorator_t>);

            static_assert(algorithm::detail::is_task_decorator_v<decorator_t>);
            static_assert(algorithm::detail::is_computation_type_v<decorator_t, computation_type::transform>);
            static_assert(algorithm::detail::get_access_type<decorator_t>() == access_type::one_to_one);

            auto seq = transform<class add>(q, begin(buf_in), end(buf_in), begin(buf_out), add_5) |
                       transform<class mul>(q, begin(buf_in), end(buf_in), begin(buf_out), mul_3);

            seq();

            THEN("kernels get fused and the result is 18")
            {
                constexpr auto num_actions = decltype(seq.get_sequence())::num_actions;
                static_assert(num_actions == 1);

                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<18>(next(begin(r)), prev(end(r))));
            }
        }
    }
}