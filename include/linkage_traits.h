#ifndef LINKAGE_TRAITS_H
#define LINKAGE_TRAITS_H

#include "computation_type_traits.h"
#include "packaged_task_traits.h"
#include "t_joint.h"

namespace celerity::algorithm::traits
{
    template <typename T>
    inline constexpr bool is_source_v = computation_type_of_v<T, detail::computation_type::generate>;

    template <typename T>
    inline constexpr bool is_linkable_source_v = is_partially_packaged_task_v<T> &&
                                                     stage_requirement_v<T> == detail::stage_requirement::output;

    template <typename T>
    inline constexpr bool is_linkable_sink_v = is_partially_packaged_task_v<T> &&stage_requirement_v<T> == detail::stage_requirement::input;

    template <typename Sink, typename Source>
    inline constexpr bool has_transiently_linkable_first_input_v = is_linkable_sink_v<Sink> && packaged_task_traits<Sink>::template access_type<Source> == detail::access_type::one_to_one;

    template <typename T>
    inline constexpr bool has_transiently_linkable_second_input_v =
        is_linkable_sink_v<T> &&computation_type_of_v<T, detail::computation_type::zip> &&second_input_access_type_v<T> == detail::access_type::one_to_one;

    template <typename T>
    constexpr inline bool first_input_stage_completed_v =
        is_packaged_task_v<
            T> ||
        (computation_type_of_v<T, algorithm::detail::computation_type::zip> && !std::is_void_v<typename extended_packaged_task_traits<T, algorithm::detail::computation_type::zip>::second_input_iterator_type>) || stage_requirement_v<T> == detail::stage_requirement::output;

    template <typename T>
    constexpr inline bool is_internally_linked_v = !is_t_joint_v<T> || first_input_stage_completed_v<T>;

} // namespace celerity::algorithm::traits

#endif // LINKAGE_TRAITS_H