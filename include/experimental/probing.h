#ifndef CELERITY_HLA_PROBING_H
#define CELERITY_HLA_PROBING_H

#include "concepts.h"
#include "../accessor_type.h"
#include "../celerity_helper.h"
#include "accessor_proxies.h"

#include "traits.h"

namespace celerity::hla::experimental
{
    template <typename T>
    concept CallableObject = celerity::algorithm::traits::has_call_operator_v<std::remove_cv_t<T>>;

    template <typename F, size_t Idx, typename ValueType, size_t Rank>
    constexpr inline auto get_access_concept()
    {
        if constexpr (is_invocable_using_probes_v<F, Idx, slice_probe<ValueType>>)
        {
            return algorithm::detail::access_type::slice;
        }
        else if constexpr (is_invocable_using_probes_v<F, Idx, block_probe<ValueType, Rank>>)
        {
            return algorithm::detail::access_type::chunk;
        }
        else
        {
            return algorithm::detail::access_type::invalid;
        }
    }

    template <size_t Arity, size_t Idx, typename F, typename T>
    constexpr auto probing_invoke(F f, T probe)
    {
        if constexpr (Arity == 1)
        {
            static_assert(Idx == 0);
            return std::invoke(f, probe);
        }
        else if constexpr (Arity == 2)
        {
            static_assert(Idx == 0 || Idx == 1);

            if constexpr (Idx == 0)
            {
                return std::invoke(f, probe, concrete_inactive_probe<typename T::value_type>{});
            }
            else
            {
                return std::invoke(f, concrete_inactive_probe<typename T::value_type>{}, probe);
            }
        }
        else
        {
            static_assert(std::is_void_v<T>, "only unary and binary functions supported");
        }
    }

    template <size_t Arity, size_t Idx, size_t Rank, typename ValueType, typename F>
    auto create_slice_proxy_factory_and_range_mapper(F f)
    {
        const auto probe = [&]() {
            try
            {
                probing_invoke<Arity, Idx>(f, slice_probe<ValueType>{});
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

        return std::tuple{factory, celerity::access::slice<Rank>{static_cast<size_t>(probe.get_dim())}};
    }

    template <int Rank, size_t... Is>
    auto create_neighbourhood_range_mapper(cl::sycl::range<Rank> range, std::index_sequence<Is...>)
    {
        return celerity::access::neighborhood<Rank>{range[Is]...};
    }

    template <size_t Arity, size_t Idx, size_t Rank, typename ValueType, typename F>
    auto create_block_proxy_factory_and_range_mapper(F f)
    {
        const auto probe = [&]() {
            try
            {
                probing_invoke<Arity, Idx>(f, block_probe<ValueType, Rank>{});
            }
            catch (const block_probe<ValueType, Rank> &p)
            {
                return p;
            }

            assert(false && "unable to probe");
            return block_probe<ValueType, Rank>{};
        }();

        const auto factory = [probe]<typename Acc, typename... Args>(Acc acc, Args && ... args)
        {
            return block<Acc>(probe, acc, std::forward<Args>(args)...);
        };

        const auto range_mapper = create_neighbourhood_range_mapper(probe.size(), std::make_index_sequence<Rank>{});

        return std::tuple{factory, range_mapper};
    }

    template <size_t Idx, size_t Rank, typename ValueType, typename F>
    auto create_proxy_factory_and_range_mapper(F f)
    {
        using celerity::algorithm::detail::access_type;
        using namespace celerity::access;

        static constexpr auto arity = kernel_arity_v<F, ValueType>;

        if constexpr (get_access_concept<F, Idx, ValueType, Rank>() == access_type::slice)
        {
            return create_slice_proxy_factory_and_range_mapper<arity, Idx, Rank, ValueType>(f);
        }
        else if constexpr (get_access_concept<F, Idx, ValueType, Rank>() == access_type::chunk)
        {
            return create_block_proxy_factory_and_range_mapper<arity, Idx, Rank, ValueType>(f);
        }
        else
        {
            const auto factory = [](auto acc, auto...) { return acc; };

            return std::tuple{factory, one_to_one<Rank>()};
        }
    }

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_PROBING_H