#ifndef CELERITY_HLA_KERNEL_TRAITS_H
#define CELERITY_HLA_KERNEL_TRAITS_H

#include "probing.h"

namespace celerity::hla::experimental
{
    template <typename F, typename Rank1, typename Arg1>
    struct unary_kernel_result : std::invoke_result<F, probe_type_t<0, Rank1::value, Arg1, F>>
    {
    };

    template <typename F, typename Rank1, typename Arg1>
    using unary_kernel_result_t = typename unary_kernel_result<F, Rank1, Arg1>::type;

    template <typename F, typename Rank1, typename Arg1, typename Rank2, typename Arg2>
    struct binary_kernel_result : std::invoke_result<F, probe_type_t<0, Rank1::value, Arg1, F>, probe_type_t<1, Rank2::value, Arg2, F>>
    {
    };

    template <typename F, typename Rank1, typename Arg1, typename Rank2, typename Arg2>
    using binary_kernel_result_t = typename binary_kernel_result<F, Rank1, Arg1, Rank2, Arg2>::type;

    template <typename F, typename... Args>
    constexpr auto get_kernel_result_type()
    {
        if constexpr (sizeof...(Args) == 2)
        {
            return typename unary_kernel_result<F, Args...>::type{};
        }
        else
        {
            return typename binary_kernel_result<F, Args...>::type{};
        }
    }

    template <typename F, typename... Args>
    using kernel_result_t = decltype(get_kernel_result_type<F, Args...>());

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_KERNEL_TRAITS_H