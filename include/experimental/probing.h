#ifndef CELERITY_HLA_PROBING_H
#define CELERITY_HLA_PROBING_H

#include "concepts.h"
#include "../accessor_type.h"
#include "../celerity_helper.h"

namespace celerity::hla::experimental
{
    template <typename T>
    concept CallableObject = celerity::algorithm::traits::has_call_operator_v<std::remove_cv_t<T>>;

    template <typename F, size_t Rank, size_t Idx, typename T>
    struct is_invocable_using_probes;

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 1, 0, T>
        : std::bool_constant<std::is_invocable_v<F, T>>
    {
    };

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 2, 0, T>
        : std::bool_constant<std::is_invocable_v<F, T, inactive_probe_t>>
    {
    };

    template <typename F, typename T>
    struct is_invocable_using_probes<F, 2, 1, T>
        : std::bool_constant<std::is_invocable_v<F, inactive_probe_t, T>>
    {
    };

    template <typename F, size_t Arity, size_t Idx, typename T>
    constexpr inline bool is_invocable_using_probes_v = is_invocable_using_probes<F, Arity, Idx, T>::value;

    template <typename F, size_t Arity, size_t Idx, typename ValueType>
    constexpr inline auto get_access_concept()
    {
        if constexpr (is_invocable_using_probes_v<F, Arity, Idx, slice_probe<ValueType>>)
        {
            return celerity::algorithm::detail::access_type::slice;
        }
        else
        {
            return celerity::algorithm::detail::access_type::invalid;
        }
    }

    template <typename F, size_t Arity, size_t Idx, size_t Rank, typename ValueType>
    auto create_proxy_factory_and_range_mapper(F f)
    {
        if constexpr (get_access_concept<F, Arity, Idx, ValueType>() == celerity::algorithm::detail::access_type::slice)
        {
            const auto probe = [&]() {
                try
                {
                    std::invoke(f, slice_probe<ValueType>{});
                }
                catch (const slice_probe<ValueType> &p)
                {
                    return p;
                }

                assert(false && "unable to probe");
                return slice_probe<ValueType>{};
            }();

            const auto factory = [probe]<typename Acc, typename... Args>(Acc acc, Args && ... args)
            {
                return slice<Acc>(probe, acc, std::forward<Args>(args)...);
            };

            return std::tuple{factory, celerity::access::slice<Rank>(probe.get_dim())};
        }
        else
        {
            const auto factory = [](auto acc, auto...) { return acc; };

            return std::tuple{factory, celerity::access::one_to_one<Rank>()};
        }
    }
} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_PROBING_H