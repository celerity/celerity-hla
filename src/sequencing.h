#ifndef SEQUENCING_H
#define SEQUENCING_H

#include "linkage.h"
#include "fusion.h"
#include "require.h"

namespace celerity::algorithm
{

template <typename T,
          require_one<detail::is_packaged_task_v<T>, detail::is_packaged_task_sequence_v<T>> = yes>
auto operator|(T lhs, distr_queue q)
{
    if constexpr (detail::is_packaged_task_v<T> || (detail::is_packaged_task_sequence_v<T> && size_v<T> == 1))
    {
        return std::invoke(lhs, q);
    }
    else
    {
        auto r = std::invoke(lhs, q);
        return std::get<std::tuple_size_v<decltype(r)> - 1>(r);
    }
}

template <typename T, require<detail::is_partially_packaged_task_v<T>,
                              detail::stage_requirement_v<T> == stage_requirement::output> = yes>
auto operator|(T lhs, distr_queue q)
{
    return terminate(sequence(lhs)) | q;
}

template <typename T, require<is_sequence_v<T>,
                              detail::is_partially_packaged_task_v<last_element_t<T>>,
                              detail::stage_requirement_v<last_element_t<T>> == stage_requirement::output> = yes>
auto operator|(T lhs, distr_queue q)
{
    return fuse(terminate(lhs)) | q;
}

inline auto submit_to(celerity::distr_queue q)
{
    return q;
}

} // namespace celerity::algorithm

#endif // SEQUENCING_H