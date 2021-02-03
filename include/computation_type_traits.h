#ifndef COMPUTATION_TYPE_TRAITS_H
#define COMPUTATION_TYPE_TRAITS_H

#include <type_traits>

#include "computation_type.h"
#include "packaged_task_traits.h"

namespace celerity::hla::traits
{

    template <typename T, detail::computation_type Type>
    struct computation_type_of : std::bool_constant<packaged_task_traits<T>::computation_type == Type>
    {
    };

    template <typename T, detail::computation_type Type>
    constexpr inline bool computation_type_of_v = computation_type_of<T, Type>::value;

    template <typename T, std::enable_if_t<computation_type_of_v<T, detail::computation_type::transform>, int> = 0>
    constexpr detail::access_type get_access_type()
    {
        return packaged_task_traits<T>::access_type;
    }

    template <typename T, typename... InputTypes>
    inline constexpr auto access_type_v = packaged_task_traits<T>::template access_type<InputTypes...>;

    template <typename T, std::enable_if_t<computation_type_of_v<T, detail::computation_type::zip>, int> = 0>
    constexpr detail::access_type get_first_access_type()
    {
        return T::first_access_type;
    }

    template <typename T, std::enable_if_t<computation_type_of_v<T, detail::computation_type::zip>, int> = 0>
    constexpr detail::access_type get_second_access_type()
    {
        return T::second_access_type;
    }

    template <typename T, std::enable_if_t<(!is_packaged_task_v<T> && !is_partially_packaged_task_v<T>) || computation_type_of_v<T, detail::computation_type::generate>, int> = 0>
    constexpr detail::access_type get_access_type()
    {
        return detail::access_type::invalid;
    }

} // namespace celerity::hla::traits

#endif // COMPUTATION_TYPE_TRAITS_H