#ifndef COMPUTATION_TYPE_H
#define COMPUTATION_TYPE_H

#include "kernel_traits.h"

namespace celerity::algorithm
{

enum class computation_type
{
    generate,
    transform,
    reduce,
    zip,
    none
};

namespace detail
{
template <typename T, std::enable_if_t<detail::is_task_decorator_v<T>, int> = 0>
constexpr computation_type get_computation_type()
{
    return T::computation_type;
}

template <typename T, std::enable_if_t<!detail::is_task_decorator_v<T>, int> = 0>
constexpr computation_type get_computation_type()
{
    return computation_type::none;
}

template <typename T, computation_type Type>
struct is_computation_type : std::bool_constant<detail::is_task_decorator_v<T> && get_computation_type<T>() == Type>
{
};

template <typename T, computation_type Type>
constexpr inline bool is_computation_type_v = is_computation_type<T, Type>::value;
} // namespace detail

} // namespace celerity::algorithm

#endif // COMPUTATION_TYPE_H