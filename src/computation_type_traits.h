#ifndef COMPUTATION_TYPE_TRAITS_H
#define COMPUTATION_TYPE_TRAITS_H

#include <type_traits>

#include "computation_type.h"
#include "decorator_traits.h"

namespace celerity::algorithm::detail
{

template <typename T>
constexpr computation_type get_computation_type()
{
    if constexpr (detail::is_packaged_task_v<T>)
    {
        return T::computation_type;
    }
    else
    {
        return computation_type::none;
    }
}

template <typename T, computation_type Type>
struct computation_type_of : std::bool_constant<get_computation_type<T>() == Type>
{
};

template <typename T, computation_type Type>
constexpr inline bool computation_type_of_v = computation_type_of<T, Type>::value;

template <typename T, std::enable_if_t<detail::computation_type_of_v<T, computation_type::transform>, int> = 0>
constexpr access_type get_access_type()
{
    return T::access_type;
}

template <typename T, std::enable_if_t<detail::computation_type_of_v<T, computation_type::zip>, int> = 0>
constexpr access_type get_first_access_type()
{
    return T::first_access_type;
}

template <typename T, std::enable_if_t<detail::computation_type_of_v<T, computation_type::zip>, int> = 0>
constexpr access_type get_second_access_type()
{
    return T::second_access_type;
}

template <typename T, std::enable_if_t<!detail::is_packaged_task_v<T> || detail::computation_type_of_v<T, computation_type::generate>, int> = 0>
constexpr access_type get_access_type()
{
    return access_type::invalid;
}

} // namespace celerity::algorithm::detail

#endif // COMPUTATION_TYPE_TRAITS_H