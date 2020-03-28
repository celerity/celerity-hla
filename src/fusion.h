#ifndef FUSION_H
#define FUSION_H

#include "fusion_traits.h"
#include "computation_type_traits.h"

#include "packaged_tasks/packaged_generate.h"
#include "packaged_tasks/packaged_transform.h"
#include "packaged_tasks/packaged_zip.h"

#include "task.h"
#include "t_joint.h"

#include "require.h"

namespace celerity::algorithm
{

namespace detail
{
template <typename... Actions>
auto fuse(const sequence<Actions...> &s);

template <typename ExecutionPolicyA, typename KernelA, typename ExecutionPolicyB, typename KernelB>
auto fuse(task_t<ExecutionPolicyA, KernelA> a, task_t<ExecutionPolicyB, KernelB> b)
{
    using new_execution_policy = named_distributed_execution_policy<
        indexed_kernel_name_t<fused<ExecutionPolicyA, ExecutionPolicyB>>>;

    using kernel_type = std::invoke_result_t<decltype(a.get_sequence()), handler &>;
    using item_type = traits::arg_type_t<kernel_type, 0>;

    auto seq = a.get_sequence() | b.get_sequence();

    auto f = [=](handler &cgh) {
        const auto kernels = sequence(std::invoke(seq, cgh));

        return [=](item_type item) {
            kernels(item);
        };
    };

    return task<new_execution_policy>(f);
}

//
//
// [a] computes first input of c
//   \
//    +----{c} computes output of c
//   /
// [b] computes second input of c
template <typename ExecutionPolicyA, typename KernelA,
          typename ExecutionPolicyB, typename KernelB,
          typename ExecutionPolicyC, typename KernelC>
auto fuse(task_t<ExecutionPolicyA, KernelA> a,
          task_t<ExecutionPolicyB, KernelB> b,
          task_t<ExecutionPolicyC, KernelC> c)
{
    using new_execution_policy = named_distributed_execution_policy<
        indexed_kernel_name_t<fused<fused<ExecutionPolicyA, ExecutionPolicyB>, ExecutionPolicyC>>>;

    using kernel_type = std::invoke_result_t<decltype(a.get_sequence()), handler &>;
    using item_type = traits::arg_type_t<kernel_type, 0>;

    auto seq_a = a.get_sequence();
    auto seq_b = b.get_sequence();
    auto seq_c = c.get_sequence();

    auto f = [=](handler &cgh) {
        const auto kernels_a = sequence(std::invoke(seq_a, cgh));
        const auto kernels_b = sequence(std::invoke(seq_b, cgh));
        const auto kernels_c = sequence(std::invoke(seq_c, cgh));

        return [=](item_type item) {
            kernels_a(item);
            // data[0] = result of a
            // data[1] = empty

            // switch item context so that
            // the b-kernels write to the
            // second data store
            item.switch_data();
            // data[0] = empty
            // data[1] = result of a

            kernels_b(item);
            // data[0] = result of b
            // data[1] = result of a

            // switch back to normal
            // data[0] = result of a
            // data[1] = result of b
            item.switch_data();

            kernels_c(item);
            // result of c written to buffer
        };
    };

    return task<new_execution_policy>(f);
}

//
//
// [ ]
//   \
//    +----{c} computes output of c
//   /
// [b] computes second (right) input of c
template <typename ExecutionPolicyB, typename KernelB,
          typename ExecutionPolicyC, typename KernelC>
auto fuse_right(task_t<ExecutionPolicyB, KernelB> b,
                task_t<ExecutionPolicyC, KernelC> c)
{
    using new_execution_policy = named_distributed_execution_policy<
        indexed_kernel_name_t<fused<ExecutionPolicyB, ExecutionPolicyC>>>;

    using kernel_type = std::invoke_result_t<decltype(b.get_sequence()), handler &>;
    using item_type = traits::arg_type_t<kernel_type, 0>;

    auto seq_b = b.get_sequence();
    auto seq_c = c.get_sequence();

    auto f = [=](handler &cgh) {
        const auto kernels_b = sequence(std::invoke(seq_b, cgh));
        const auto kernels_c = sequence(std::invoke(seq_c, cgh));

        return [=](item_type item) {
            // switch item context so that
            // the b-kernels write to the
            // second data store
            item.switch_data();
            // data[0] = empty
            // data[1] = empty

            kernels_b(item);
            // data[0] = result of b
            // data[1] = empty

            // switch back to normal
            // data[0] = empty
            // data[1] = result of b
            item.switch_data();

            kernels_c(item);
            // data[0] = result of c
            // data[1] = empty
        };
    };

    return task<new_execution_policy>(f);
}

template <typename T, typename U,
          require<traits::are_fusable_v<T, U>,
                  traits::computation_type_of_v<T, computation_type::transform>,
                  !traits::is_t_joint_v<U>> = yes>
auto fuse(T lhs, U rhs)
{
    return package_transform<access_type::one_to_one>(fuse(lhs.get_task(), rhs.get_task()),
                                                      lhs.get_in_beg(),
                                                      lhs.get_in_end(),
                                                      rhs.get_out_beg());

    // Results in a linker error. Not sure why -> need further clarification from philip/peter
    //
    // return package_transform<access_type::one_to_one, true>(task<new_execution_policy>(seq),
    //                                                     lhs.get_in_beg(),
    //                                                     lhs.get_in_end(),
    //
}

template <typename T, typename U,
          require<traits::are_fusable_v<T, U>,
                  traits::computation_type_of_v<T, computation_type::generate>,
                  !traits::is_t_joint_v<U>> = yes>
auto fuse(T lhs, U rhs)
{
    using output_value_type = typename traits::packaged_task_traits<U>::output_value_type;

    auto out_beg = rhs.get_out_beg();
    auto out_end = end(out_beg.get_buffer());

    return package_generate<output_value_type>(fuse(lhs.get_task(), rhs.get_task()), out_beg, out_end);
}

template <typename T, typename U,
          require<traits::is_packaged_task_v<T>,
                  traits::is_t_joint_v<U>> = yes>
auto fuse(T lhs, U rhs)
{
    using namespace detail;

    constexpr auto first_input_access_type = traits::packaged_task_traits<U>::access_type;
    constexpr auto second_input_access_type = traits::extended_packaged_task_traits<U, computation_type::zip>::second_input_access_type;

    auto fused_secondary = fuse(rhs.get_secondary());
    using secondary_input_sequence = decltype(fused_secondary);

    // both primary and secondary input sequence fusable
    if constexpr (traits::has_transient_input_v<U> &&
                  traits::has_transient_second_input_v<U>)
    {
        const auto fused = fuse(lhs.get_task(),
                                get_last_element(fused_secondary).get_task(),
                                rhs.get_task().get_task());

        auto lhs_out_beg = lhs.get_out_beg();
        auto lhs_out_end = end(lhs_out_beg.get_buffer());
        auto secondary_out_beg = get_last_element(fused_secondary).get_out_beg();

        auto zip = package_zip<first_input_access_type, second_input_access_type>(fused,
                                                                                  lhs_out_beg,
                                                                                  lhs_out_end,
                                                                                  secondary_out_beg,
                                                                                  rhs.get_task().get_out_beg());

        if constexpr (traits::size_v<secondary_input_sequence> == 1)
        {
            return sequence(zip);
        }
        else
        {
            return sequence(t_joint{zip, remove_last_element(fused_secondary)});
        }
    }
    // only primary input is fusable
    // TODO: Not enabled yet -> are_fusable_v<T, U> needs adaption
    else if constexpr (traits::has_transient_input_v<U> &&
                       !traits::has_transient_second_input_v<U>)
    {
        const auto fused = fuse(lhs.get_task(),
                                rhs.get_task().get_task());

        auto lhs_out_beg = lhs.get_out_beg();
        auto lhs_out_end = end(lhs_out_beg.get_buffer());
        auto secondary_out_beg = get_last_element(fused_secondary).get_out_beg();

        auto zip = package_zip<first_input_access_type, second_input_access_type>(fused,
                                                                                  lhs_out_beg,
                                                                                  lhs_out_end,
                                                                                  secondary_out_beg,
                                                                                  rhs.get_task().get_out_beg());

        return sequence(t_joint{zip, fused_secondary});
    }
    // only secondary input fusable
    // TODO: Not enabled yet -> are_fusable_v<T, U> needs adaption
    else if constexpr (!traits::has_transient_input_v<U> &&
                       traits::has_transient_second_input_v<U>)
    {
        const auto fused = fuse_right(get_last_element(fused_secondary).get_task(),
                                      rhs.get_task().get_task());

        auto lhs_out_beg = lhs.get_out_beg();
        auto lhs_out_end = end(lhs_out_beg.get_buffer());
        auto secondary_out_beg = get_last_element(fused_secondary).get_out_beg();

        auto zip = package_zip<first_input_access_type, second_input_access_type>(fused,
                                                                                  lhs_out_beg,
                                                                                  lhs_out_end,
                                                                                  secondary_out_beg,
                                                                                  rhs.get_task().get_out_beg());

        if constexpr (traits::size_v<secondary_input_sequence> == 1)
        {
            return sequence(lhs, zip);
        }
        else
        {
            return sequence(lhs, t_joint{zip, remove_last_element(fused_secondary)});
        }
    }
    // none is fusable
    else
    {
        return sequence(lhs, rhs);
    }
}

template <typename T,
          require<traits::is_t_joint_v<T>> = yes>
auto fuse(T joint)
{
    using namespace detail;

    constexpr auto first_input_access_type = traits::packaged_task_traits<T>::access_type;
    constexpr auto second_input_access_type = traits::extended_packaged_task_traits<T, computation_type::zip>::second_input_access_type;

    auto fused_secondary = fuse(joint.get_secondary());
    using secondary_input_sequence = decltype(fused_secondary);

    // secondary sequence fusable
    if constexpr (traits::has_transient_second_input_v<T>)
    {
        const auto fused = fuse_right(get_last_element(fused_secondary).get_task(),
                                      joint.get_task().get_task());

        auto in_beg = joint.get_task().get_in_beg();
        auto in_end = joint.get_task().get_in_end();
        auto secondary_out_beg = get_last_element(fused_secondary).get_out_beg();
        auto out_beg = joint.get_task().get_out_beg();

        auto zip = package_zip<first_input_access_type, second_input_access_type>(fused,
                                                                                  in_beg,
                                                                                  in_end,
                                                                                  secondary_out_beg,
                                                                                  out_beg);
        if constexpr (traits::size_v<secondary_input_sequence> == 1)
        {
            return zip;
        }
        else
        {
            return t_joint{zip, remove_last_element(fused_secondary)};
        }
    }
    // non-fusable
    else
    {
        return joint;
    }
}

template <typename T,
          require<traits::is_packaged_task_v<T>,
                  !traits::is_t_joint_v<T>> = yes>
auto fuse(T task)
{
    return task;
}

} // namespace detail

template <typename T, typename U,
          require_one<traits::are_fusable_v<T, U>,
                      traits::is_packaged_task_v<T> && traits::is_t_joint_v<U>> = yes>
auto operator|(T lhs, U rhs)
{
    using namespace detail;
    return sequence(fuse(lhs, rhs));
}

template <typename T, typename U,
          require<traits::is_packaged_task_v<T>,
                  traits::is_packaged_task_v<U>,
                  !traits::is_t_joint_v<U>,
                  !traits::are_fusable_v<T, U>> = yes>
auto operator|(T lhs, U rhs)
{
    using namespace detail;
    return sequence(lhs, rhs);
}

template <typename T, typename U,
          require<traits::is_packaged_task_sequence_v<T>,
                  traits::is_packaged_task_v<U>> = yes>
auto operator|(T lhs, U rhs)
{
    using namespace detail;
    return apply_append(lhs, rhs);
}

namespace detail
{

template <typename... Actions, size_t... Is>
auto fuse(const sequence<Actions...> &s, std::index_sequence<Is...>)
{
    const auto &actions = s.actions();
    return (... | (std::get<Is>(actions)));
}

template <typename... Actions>
auto fuse(const sequence<Actions...> &s)
{
    if constexpr (sizeof...(Actions) == 1)
    {
        if constexpr (!traits::is_t_joint_v<traits::first_element_t<sequence<Actions...>>>)
        {
            return s;
        }
        else
        {
            return sequence(fuse(detail::get_first_element(s)));
        }
    }
    else
    {
        return fuse(s, std::make_index_sequence<sizeof...(Actions)>{});
    }
}

} // namespace detail

} // namespace celerity::algorithm

#endif // FUSION_H`