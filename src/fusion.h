#ifndef FUSION_H
#define FUSION_H

#include "decoration.h"

namespace celerity::algorithm
{

template <typename T, int Rank>
struct transient_buffer : buffer<T, Rank>
{
};

template <typename T>
constexpr auto is_simple_transform_decorator_v = detail::is_computation_type_v<T, computation_type::transform> &&
                                                 detail::get_access_type<T>() == access_type::one_to_one;

// TODO:
//
// Does not work like that.
//
// move buffer identification to compile time using some kind of indexing
// and only do sanity checks at runtime. Needs to be done at runtime, otherwise
// we can not fuse kernels as invoking function pointers (to fused, type-erased kernels)
// is not permitted in device code.
//
// Idea is to tag buffers with ids before computation to tell them apart. Then do the same
// distinction of fusion cases as below.
//
template <typename T, typename U, std::enable_if_t<is_simple_transform_decorator_v<T> && is_simple_transform_decorator_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    // GENERAL REQUIREMENTS
    //
    // both operands are simple transformations
    //
    // CASE 1:
    //
    // both read from the same buffer and write to the same buffer
    // and access only one element each
    //
    // FUSION OPPORTUNITY:
    //
    // we can execute both transformations in sequence in the same kernel
    //
    if (lhs.get_in_beg().get_buffer().get_id() ==
            rhs.get_in_beg().get_buffer().get_id() &&
        lhs.get_out_iterator().get_buffer().get_id() ==
            rhs.get_out_iterator().get_buffer().get_id())
    {
        using task_t = decltype(lhs.get_task());
        using execution_policy_t = typename task_t::execution_policy_type;

        auto f_a = lhs.get_computation_functor();
        auto f_b = rhs.get_computation_functor();

        return decorate_transform<access_type::one_to_one>([f_a, f_b](auto in_beg, auto in_end, auto out_in) {
            auto task_a = std::invoke(f_a, in_beg, in_end, out_in);
            auto task_b = std::invoke(f_b, in_beg, in_end, out_in);

            return task<execution_policy_t>(
                task_a.get_sequence() | task_b.get_sequence());
        },
                                                           lhs.get_in_beg(), lhs.get_in_end(), lhs.get_out_iterator());
    }

    //return sequence(lhs, rhs);
}

} // namespace celerity::algorithm

#endif // FUSION_H