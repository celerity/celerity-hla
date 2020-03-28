#ifndef FUSION_TRAITS_H
#define FUSION_TRAITS_H

#include "packaged_task_traits.h"
#include "t_joint.h"
#include "transient.h"

namespace celerity::algorithm::traits
{

template <typename T>
inline constexpr bool has_transient_output_v = is_transient_v<typename packaged_task_traits<T>::output_iterator_type>;

template <typename T>
inline constexpr bool has_transient_input_v = is_transient_v<typename packaged_task_traits<T>::input_iterator_type>;

template <typename T>
inline constexpr bool has_transient_second_input_v = is_transient_v<typename extended_packaged_task_traits<T, detail::computation_type::zip>::second_input_iterator_type>;

template <typename T>
inline constexpr bool is_fusable_source_v = is_packaged_task_v<T> &&has_transient_output_v<T> && !is_t_joint_v<T>;

template <typename T>
inline constexpr bool is_fusable_sink_v = (is_packaged_task_v<T> && has_transient_input_v<T>) || (is_t_joint_v<T> && has_transient_second_input_v<T>);

template <typename T, typename U>
inline constexpr bool are_fusable_v = is_fusable_source_v<T> &&is_fusable_sink_v<U>;

} // namespace celerity::algorithm::traits

#endif // FUSION_TRAITS_H