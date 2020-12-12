#ifndef CELERITY_HLA_TRAITS_H
#define CELERITY_HLA_TRAITS_H

#include <type_traits>

#include "../accessor_type.h"

#include "inactive_probe.h"


namespace celerity::hla::experimental
{
    template <typename F, typename T, size_t Max, typename... Args>
    constexpr bool is_kernel()
    {
        if constexpr (sizeof...(Args) > Max)
        {
            return false;
        }
        else if constexpr (!std::is_invocable_v<F, Args...>)
        {
            return is_kernel<F, T, Max, Args..., T>();
        }
        else
        {
            return sizeof...(Args);
        }
    }

    template <typename F, typename T>
    constexpr inline bool is_kernel_v = is_kernel<F, concrete_inactive_probe<T>, 2>();

    template <typename F, typename T, size_t Max, typename... Args>
    constexpr size_t get_kernel_arity()
    {
        if constexpr (!std::is_invocable_v<F, Args...>)
        {
            static_assert(sizeof...(Args) < Max, "not incovable with < Max probes");
            return get_kernel_arity<F, T, Max, Args..., T>();
        }
        else
        {
            return sizeof...(Args);
        }
    }

    template <typename F, typename T>
    constexpr size_t get_kernel_arity()
    {
        return get_kernel_arity<F, concrete_inactive_probe<T>, 2>();
    }

    template <typename KernelType, typename ValueType>
    constexpr inline auto kernel_arity_v = get_kernel_arity<KernelType, ValueType>();

    template <typename F, size_t Rank, size_t Idx, typename T>
    struct is_invocable_using_probes;

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 1, 0, T>
        : std::bool_constant<std::is_invocable_v<F, T>>
    {
    };

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 2, 0, T>
        : std::bool_constant<std::is_invocable_v<F, T, concrete_inactive_probe<typename T::value_type>>>
    {
    };

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 2, 1, T>
        : std::bool_constant<std::is_invocable_v<F, concrete_inactive_probe<typename T::value_type>, T>>
    {
    };

    template <typename F, size_t Idx, typename T>
    constexpr inline bool is_invocable_using_probes_v = is_invocable_using_probes<F, kernel_arity_v<F, typename T::value_type>, Idx, T>::value;
} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_TRAITS_H