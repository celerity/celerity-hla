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

    template <typename First, typename...>
    struct first
    {
        using type = First;
    };

    template <typename... Args>
    using first_t = typename first<Args...>::type;

    template <typename, typename Second, typename...>
    struct second
    {
        using type = Second;
    };

    template <typename... Args>
    using second_t = typename second<Args...>::type;

    template <typename F, typename... Args>
    constexpr bool is_kernel()
    {
        static_assert(sizeof...(Args) <= 2);

        if constexpr (sizeof...(Args) == 1)
        {
            return std::is_invocable_v<F, concrete_inactive_probe<Args>...> ||
                   std::is_invocable_v<F, Args...>;
        }
        else
        {
            return std::is_invocable_v<F, concrete_inactive_probe<Args>...> ||
                   std::is_invocable_v<F, Args...> ||
                   std::is_invocable_v<F, concrete_inactive_probe<first_t<Args...>>, second_t<Args...>> ||
                   std::is_invocable_v<F, first_t<Args...>, concrete_inactive_probe<second_t<Args...>>>;
        }
    }

    template <typename F, typename... Args>
    constexpr inline bool is_kernel_v = is_kernel<F, Args...>();

    template <typename F, typename T, size_t Max, typename... Args>
    constexpr size_t get_kernel_arity()
    {
        if constexpr (!std::is_invocable_v<F, Args...>)
        {
            static_assert(sizeof...(Args) <= Max, "not incovable with < Max probes");
            return get_kernel_arity<F, T, Max, Args..., T>();
        }
        else
        {
            return sizeof...(Args);
        }
    }

    // template <typename F, typename T>
    // constexpr size_t get_kernel_arity()
    // {
    //     if constexpr (is_kernel_v<F, T>)
    //     {
    //         return 1;
    //     }
    //     else if constexpr (is_kernel_v<F, T, T)
    // }

    //template <typename KernelType, typename ValueType>
    //constexpr inline auto kernel_arity_v = get_kernel_arity<KernelType, ValueType>();

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

    template <typename F, size_t Arity, size_t Idx, typename T>
    constexpr inline bool is_invocable_using_probes_v = is_invocable_using_probes<F, Arity, Idx, T>::value;
} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_TRAITS_H