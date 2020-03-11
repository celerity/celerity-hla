#ifndef LINKAGE_TRAITS_H
#define LINKAGE_TRAITS_H

#include "packaged_task_traits.h"
#include "computation_type_traits.h"
#include "t_joint.h"

namespace celerity::algorithm
{

template <typename T>
inline constexpr bool is_source_v = detail::computation_type_of_v<T, computation_type::generate>;

template <typename T>
inline constexpr bool single_element_access_v = (detail::computation_type_of_v<T, computation_type::transform> &&
                                                 detail::access_type_v<T> == access_type::one_to_one) ||
                                                (detail::computation_type_of_v<T, computation_type::zip> &&
                                                 detail::access_type_v<T> == access_type::one_to_one &&
                                                 detail::second_input_access_type_v<T> == access_type::one_to_one);

template <typename T>
inline constexpr bool is_linkable_source_v = detail::is_partially_packaged_task_v<T> &&
                                                 detail::stage_requirement_v<T> == stage_requirement::output;

template <typename T>
inline constexpr bool is_linkable_sink_v = detail::is_partially_packaged_task_v<T> &&
                                               detail::stage_requirement_v<T> == stage_requirement::input;

template <typename T>
inline constexpr bool is_transiently_linkable_source_v = is_linkable_source_v<T> && !detail::is_t_joint_v<T>;

template <typename T>
inline constexpr bool is_transiently_linkable_sink_v = is_linkable_sink_v<T> &&single_element_access_v<T> && !detail::is_t_joint_v<T>;

} // namespace celerity::algorithm

#endif // LINKAGE_TRAITS_H