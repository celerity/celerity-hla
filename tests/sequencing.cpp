#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include "utils.h"

#include "../include/sequencing.h"
#include "../include/actions.h"
#include "../include/fusion_helper.h"

#include <numeric>

using namespace celerity;
using namespace celerity::hla;
using namespace celerity::hla::traits;
using namespace celerity::hla::util;
using celerity::hla::chunk;

SCENARIO("Fusing two tasks", "[fusion::simple]")
{
    constexpr auto identity = [](int x) { return x; };

    GIVEN("buffer -> transform")
    {
        using b0 = celerity::buffer<int, 1>;
        using t0 = decltype(transform<class _00>(identity));

        using s0 = decltype(
            std::declval<b0>() |
            std::declval<t0>());

        static_assert(size_v<s0> == 2);
        static_assert(is_celerity_buffer_v<first_element_t<s0>>);
        static_assert(is_partially_packaged_task_v<last_element_t<s0>>);

        {
            using s1 = resolved_t<s0>;
            static_assert(size_v<s1> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<s1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s1>>);
        }

        {
            using s2 = linked_t<s0>;
            static_assert(size_v<s2> == 1);
            static_assert(is_partially_packaged_task_v<first_element_t<s2>>);
        }

        {
            using s3 = terminated_t<s0>;
            static_assert(size_v<s3> == 1);
            static_assert(is_packaged_task_v<first_element_t<s3>>);
        }

        {
            using s4 = fused_t<s0>;
            static_assert(size_v<s4> == 1);
            static_assert(is_packaged_task_v<first_element_t<s4>>);
        }
    }

    GIVEN("buffer -> transform -> transform")
    {
        using b0 = celerity::buffer<int, 1>;
        using t0 = decltype(transform<class _10>(identity));
        using t1 = decltype(transform<class _11>(identity));

        using s0 = decltype(
            std::declval<b0>() |
            std::declval<t0>() |
            std::declval<t1>());

        static_assert(size_v<s0> == 3);
        static_assert(is_celerity_buffer_v<first_element_t<s0>>);
        static_assert(is_partially_packaged_task_v<nth_element_t<1, s0>>);
        static_assert(is_partially_packaged_task_v<last_element_t<s0>>);

        {
            using s1 = resolved_t<s0>;
            static_assert(size_v<s1> == 3);
            static_assert(is_celerity_buffer_v<first_element_t<s1>>);
            static_assert(is_partially_packaged_task_v<nth_element_t<1, s1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s1>>);
        }

        {
            using s2 = linked_t<s0>;
            static_assert(size_v<s2> == 2);
            static_assert(is_packaged_task_v<first_element_t<s2>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s2>>);
        }

        {
            using s3 = terminated_t<s0>;
            static_assert(size_v<s3> == 2);
            static_assert(is_packaged_task_sequence_v<s3>);
        }

        {
            using s4 = fused_t<s0>;
            static_assert(size_v<s4> == 1);
            static_assert(is_packaged_task_v<first_element_t<s4>>);
        }
    }

    GIVEN("buffer -> (buffer -> zip)")
    {
        using b0 = celerity::buffer<int, 1>;
        using b1 = celerity::buffer<int, 1>;
        using z0 = decltype(transform<class _20>(std::plus<int>{}));

        using s0 = decltype(
            std::declval<b0>() |
            std::declval<z0>() << std::declval<b1>());

        {
            static_assert(size_v<s0> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<s0>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s0>>);
            static_assert(is_t_joint_v<last_element_t<s0>>);

            using ss0 = secondary_sequence_t<last_element_t<s0>>;
            static_assert(size_v<ss0> == 1);
            static_assert(is_celerity_buffer_v<first_element_t<ss0>>);
        }

        {
            using s1 = resolved_t<s0>;
            static_assert(size_v<s1> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<s1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s1>>);
            static_assert(is_t_joint_v<last_element_t<s1>>);

            using ss1 = secondary_sequence_t<last_element_t<s1>>;
            static_assert(size_v<ss1> == 1);
            static_assert(is_celerity_buffer_v<first_element_t<ss1>>);
        }

        {
            using s2 = linked_t<s0>;
            static_assert(size_v<s2> == 1);
            static_assert(is_partially_packaged_task_v<first_element_t<s2>>);
        }

        {
            using s3 = terminated_t<s0>;
            static_assert(size_v<s3> == 1);
            static_assert(is_packaged_task_v<first_element_t<s3>>);
        }

        {
            using s4 = fused_t<s0>;
            static_assert(size_v<s4> == 1);
            static_assert(is_packaged_task_v<first_element_t<s4>>);
        }
    }

    GIVEN("buffer -> transform -> (buffer -> zip)")
    {
        using b0 = celerity::buffer<int, 1>;
        using b1 = celerity::buffer<int, 1>;
        using t0 = decltype(transform<class _30>(identity));
        using z0 = decltype(transform<class _31>(std::plus<int>{}));

        using s0 = decltype(
            std::declval<b0>() |
            std::declval<t0>() |
            std::declval<z0>() << std::declval<b1>());

        {
            static_assert(size_v<s0> == 3);
            static_assert(is_celerity_buffer_v<first_element_t<s0>>);
            static_assert(is_partially_packaged_task_v<nth_element_t<1, s0>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s0>>);
            static_assert(is_t_joint_v<last_element_t<s0>>);

            using ss0 = secondary_sequence_t<last_element_t<s0>>;
            static_assert(size_v<ss0> == 1);
            static_assert(is_celerity_buffer_v<first_element_t<ss0>>);
        }

        {
            using s1 = resolved_t<s0>;
            static_assert(size_v<s1> == 3);
            static_assert(is_celerity_buffer_v<first_element_t<s1>>);
            static_assert(is_partially_packaged_task_v<nth_element_t<1, s1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s1>>);
            static_assert(is_t_joint_v<last_element_t<s1>>);

            using ss1 = secondary_sequence_t<last_element_t<s1>>;
            static_assert(size_v<ss1> == 1);
            static_assert(is_celerity_buffer_v<first_element_t<ss1>>);
        }

        {
            using s2 = linked_t<s0>;
            static_assert(size_v<s2> == 2);
            static_assert(is_packaged_task_v<first_element_t<s2>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s2>>);
            static_assert(!is_t_joint_v<last_element_t<s2>>);
        }

        {
            using s3 = terminated_t<s0>;
            static_assert(size_v<s3> == 2);
            static_assert(is_packaged_task_sequence_v<s3>);
            static_assert(!is_t_joint_v<last_element_t<s3>>);
        }

        {
            using s4 = fused_t<s0>;
            static_assert(size_v<s4> == 1);
            static_assert(is_packaged_task_v<first_element_t<s4>>);
            static_assert(!is_t_joint_v<first_element_t<s4>>);
        }
    }

    GIVEN("buffer -> (buffer -> transform -> zip)")
    {
        using b0 = celerity::buffer<int, 1>;
        using b1 = celerity::buffer<int, 1>;
        using t0 = decltype(transform<class _40>(identity));
        using z0 = decltype(transform<class _41>(std::plus<int>{}));

        using s0 = decltype(
            std::declval<b0>() |
            std::declval<z0>() << (std::declval<b1>() | std::declval<t0>()));

        {
            static_assert(size_v<s0> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<s0>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s0>>);
            static_assert(is_t_joint_v<last_element_t<s0>>);

            using ss0 = secondary_sequence_t<last_element_t<s0>>;
            static_assert(size_v<ss0> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<ss0>>);
            static_assert(is_partially_packaged_task_v<last_element_t<ss0>>);
        }

        {
            using s1 = resolved_t<s0>;
            static_assert(size_v<s1> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<s1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<s1>>);
            static_assert(is_t_joint_v<last_element_t<s1>>);

            using ss1 = secondary_sequence_t<last_element_t<s1>>;
            static_assert(size_v<ss1> == 2);
            static_assert(is_celerity_buffer_v<first_element_t<ss1>>);
            static_assert(is_partially_packaged_task_v<last_element_t<ss1>>);
        }

        {
            using s2 = linked_t<s0>;
            static_assert(size_v<s2> == 1);
            static_assert(is_partially_packaged_task_v<first_element_t<s2>>);
            static_assert(is_t_joint_v<last_element_t<s2>>);

            using ss2 = secondary_sequence_t<last_element_t<s2>>;
            static_assert(size_v<ss2> == 1);
            static_assert(is_packaged_task_v<last_element_t<ss2>>);
        }

        {
            using s3 = terminated_t<s0>;
            static_assert(size_v<s3> == 1);
            static_assert(is_packaged_task_sequence_v<s3>);
        }

        {
            using s4 = fused_t<s0>;
            static_assert(size_v<s4> == 1);
            static_assert(is_packaged_task_v<first_element_t<s4>>);
            static_assert(!is_t_joint_v<first_element_t<s4>>);
        }
    }
}