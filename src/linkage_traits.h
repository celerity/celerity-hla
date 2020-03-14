#ifndef LINKAGE_TRAITS_H
#define LINKAGE_TRAITS_H

#include "packaged_task_traits.h"
#include "computation_type_traits.h"
#include "t_joint.h"

namespace celerity::algorithm::traits
{

template <typename T>
inline constexpr bool is_source_v = computation_type_of_v<T, detail::computation_type::generate>;

template <typename T>
inline constexpr bool single_element_access_v = (computation_type_of_v<T, detail::computation_type::transform> &&
                                                 access_type_v<T> == detail::access_type::one_to_one) ||
                                                (computation_type_of_v<T, detail::computation_type::zip> &&
                                                 access_type_v<T> == detail::access_type::one_to_one &&
                                                 second_input_access_type_v<T> == detail::access_type::one_to_one);

template <typename T>
inline constexpr bool is_linkable_source_v = is_partially_packaged_task_v<T> &&
                                                 stage_requirement_v<T> == detail::stage_requirement::output;

template <typename T>
inline constexpr bool is_linkable_sink_v = is_partially_packaged_task_v<T> &&
                                               stage_requirement_v<T> == detail::stage_requirement::input;

template <typename T>
inline constexpr bool is_transiently_linkable_source_v = is_linkable_source_v<T> && !is_t_joint_v<T>;

template <typename T>
inline constexpr bool is_transiently_linkable_sink_v = is_linkable_sink_v<T> &&single_element_access_v<T>; // && !detail::is_t_joint_v<T>;

} // namespace celerity::algorithm::traits

#endif // LINKAGE_TRAITS_H