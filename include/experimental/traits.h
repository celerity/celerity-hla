#ifndef CELERITY_HLA_TRAITS_H
#define CELERITY_HLA_TRAITS_H

#include <type_traits>

#include "../accessor_type.h"

#include "inactive_probe.h"
#include "slice.h"
#include "block.h"
#include "all.h"

namespace celerity::hla::experimental
{
    struct unused
    {
    };

    template <typename...>
    struct first
    {
        using type = void;
    };

    template <typename First, typename... Tail>
    struct first<First, Tail...>
    {
        using type = First;
    };

    template <typename... Args>
    using first_t = typename first<Args...>::type;

    template <typename...>
    struct second
    {
        using type = void;
    };

    template <typename First, typename Second, typename... Tail>
    struct second<First, Second, Tail...>
    {
        using type = Second;
    };

    template <typename... Args>
    using second_t = typename second<Args...>::type;

    template <size_t Idx, typename... Args>
    using nth_t = std::conditional_t<Idx == 0, first_t<Args...>, second_t<Args...>>;

    template <typename ValueType, size_t Rank>
    struct kernel_input
    {
        using value_type = ValueType;
        static constexpr auto rank = Rank;
    };

    template <typename T>
    struct is_kernel_input : std::bool_constant<false>
    {
    };

    template <typename ValueType, size_t Rank>
    struct is_kernel_input<kernel_input<ValueType, Rank>> : std::bool_constant<true>
    {
    };

    template <typename T>
    concept KernelInput = is_kernel_input<T>::value;

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

    template <typename F, size_t Arity, size_t Idx, typename ProbeType, KernelInput... Args>
    struct is_invocable_using_probes;

    template <typename F, typename ProbeType, KernelInput Arg>
    struct is_invocable_using_probes<F, 1, 0, ProbeType, Arg>
        : std::bool_constant<std::is_invocable_v<F, ProbeType>>
    {
    };

    template <typename F, typename ProbeType, KernelInput... Args>
    struct is_invocable_using_probes<F, 2, 0, ProbeType, Args...>
        : std::bool_constant<std::is_invocable_v<F, ProbeType, concrete_inactive_probe<typename second_t<Args...>::value_type>>>
    {
        static_assert(sizeof...(Args) == 2);
    };

    template <typename F, typename ProbeType, KernelInput... Args>
    struct is_invocable_using_probes<F, 2, 1, ProbeType, Args...>
        : std::bool_constant<std::is_invocable_v<F, concrete_inactive_probe<typename first_t<Args...>::value_type>, ProbeType>>
    {
        static_assert(sizeof...(Args) == 2);
    };

    template <typename F, size_t Arity, size_t Idx, typename ProbeType, KernelInput... Args>
    constexpr inline bool is_invocable_using_probes_v = is_invocable_using_probes<F, Arity, Idx, ProbeType, Args...>::value;

    template <typename F, size_t Arity, size_t Idx, KernelInput... Args>
    constexpr inline auto get_access_concept()
    {
        static_assert(Idx < Arity, "invalid argument index");

        using nth_type = nth_t<Idx, Args...>;
        using nth_value_type = typename nth_type::value_type;
        constexpr auto nth_rank = nth_type::rank;

        if constexpr (is_invocable_using_probes_v<F, Arity, Idx, slice_probe<nth_value_type>, Args...>)
        {
            return algorithm::detail::access_type::slice;
        }
        else if constexpr (is_invocable_using_probes_v<F, Arity, Idx, block_probe<nth_value_type, nth_rank>, Args...>)
        {
            return algorithm::detail::access_type::chunk;
        }
        else if constexpr (is_invocable_using_probes_v<F, Arity, Idx, all_probe<nth_value_type, nth_rank>, Args...>)
        {
            return algorithm::detail::access_type::all;
        }
        else //if constexpr (is_invocable_using_probes_v<F, Arity, Idx, ValueType>)
        {
            return algorithm::detail::access_type::one_to_one;
        }
        // else
        // {
        //     return algorithm::detail::access_type::invalid;
        // }
    }

    template <typename F, size_t Arity, size_t Idx, KernelInput... Args>
    constexpr inline auto access_concept_v = get_access_concept<F, Arity, Idx, Args...>();

    template <typename F, size_t Arity, size_t Idx, KernelInput... Args>
    constexpr auto get_probe_type()
    {
        using celerity::algorithm::detail::access_type;
        using namespace celerity::access;

        using nth_type = nth_t<Idx, Args...>;
        using nth_value_type = typename nth_type::value_type;
        constexpr auto nth_rank = nth_type::rank;

        if constexpr (get_access_concept<F, Arity, Idx, Args...>() == access_type::slice)
        {
            (void)nth_rank;
            return slice_probe<nth_value_type>{};
        }
        else if constexpr (get_access_concept<F, Arity, Idx, Args...>() == access_type::chunk)
        {
            return block_probe<nth_value_type, nth_rank>{};
        }
        else if constexpr (get_access_concept<F, Arity, Idx, Args...>() == access_type::all)
        {
            return all_probe<nth_value_type, nth_rank>{};
        }
        else
        {
            return nth_value_type{};
        }
    }

    template <typename F, size_t Arity, size_t Idx, KernelInput... Args>
    using probe_type_t = std::decay_t<decltype(get_probe_type<F, Arity, Idx, Args...>())>;

    template <typename F, typename... Args, size_t... Is>
    constexpr auto get_kernel_result_type(std::index_sequence<Is...>)
    {
        static_assert(sizeof...(Is) == sizeof...(Args));
        return std::invoke_result_t<F, probe_type_t<F, sizeof...(Args), Is, Args...>...>{};
    }

    template <typename F, KernelInput... Args>
    using kernel_result_t = decltype(get_kernel_result_type<F, Args...>(std::index_sequence_for<Args...>{}));

    template <typename F, KernelInput... Args>
    struct kernel_traits
    {
        static constexpr auto arity = sizeof...(Args);

        static_assert(arity <= 2, "only unary and binary kernels supported");

        template <size_t Idx>
        struct argument
        {
            static constexpr algorithm::detail::access_type access_concept = access_concept_v<F, arity, Idx, Args...>;

            using probe_type = probe_type_t<F, arity, Idx, Args...>;

            using value_type = typename nth_t<Idx, Args...>::value_type;
            static constexpr auto rank = nth_t<Idx, Args...>::rank;
        };

        using result_type = kernel_result_t<F, Args...>;
    };

    template <typename T, typename... Args>
    concept Kernel = is_kernel_v<T, Args...>;

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_TRAITS_H