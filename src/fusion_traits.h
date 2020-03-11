#ifndef FUSION_TRAITS_H
#define FUSION_TRAITS_H

#include "packaged_task_traits.h"
#include "t_joint.h"
#include "transient.h"

namespace celerity::algorithm
{

template <typename T>
inline constexpr bool has_transient_output_v = is_transient_v<typename detail::packaged_task_traits<T>::output_iterator_type>;

template <typename T>
inline constexpr bool has_transient_input_v = is_transient_v<typename detail::packaged_task_traits<T>::input_iterator_type>;

template <typename T>
inline constexpr bool is_fusable_source_v = detail::is_packaged_task_v<T> &&has_transient_output_v<T> && !detail::is_t_joint_v<T>;

template <typename T>
inline constexpr bool is_fusable_sink_v = detail::is_packaged_task_v<T> &&has_transient_input_v<T> && !detail::is_t_joint_v<T>;

} // namespace celerity::algorithm

#endif // FUSION_TRAITS_H