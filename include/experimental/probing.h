#ifndef CELERITY_HLA_PROBING_H
#define CELERITY_HLA_PROBING_H

#include "concepts.h"
#include "../accessor_type.h"
#include "../celerity_helper.h"

#include "slice.h"
#include "block.h"
#include "all.h"

#include "traits.h"

namespace celerity::hla::experimental
{
    template <typename T>
    concept CallableObject = celerity::algorithm::traits::has_call_operator_v<std::remove_cv_t<T>>;

    template <size_t Idx, KernelInput... Args, typename ProbeType, typename F>
    constexpr auto probing_invoke(F f, ProbeType probe)
    {
        constexpr auto arity = sizeof...(Args);
        static_assert(Idx < arity, "invalid argument index");

        if constexpr (arity == 1)
        {
            static_assert(Idx == 0);
            return std::invoke(f, probe);
        }
        else if constexpr (arity == 2)
        {
            static_assert(Idx == 0 || Idx == 1);

            if constexpr (Idx == 0)
            {
                return std::invoke(f, probe, concrete_inactive_probe<typename second_t<Args...>::value_type>{});
            }
            else
            {
                return std::invoke(f, concrete_inactive_probe<typename first_t<Args...>::value_type>{}, probe);
            }
        }
        else
        {
            static_assert(std::is_void_v<ProbeType>, "only unary and binary functions supported");
        }
    }

    template <size_t Idx, KernelInput... Args, typename F, typename ProbeType>
    auto probe(F f, ProbeType probe)
    {
        try
        {
            probing_invoke<Idx, Args...>(f, ProbeType{});
        }
        catch (const ProbeType &p)
        {
            return p;
        }

        assert(false && "unable to probe");
        return ProbeType{};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_slice_proxy_factory_and_range_mapper(F f)
    {
        using nth_type = nth_t<Idx, Args...>;
        using probe_type = slice_probe<typename nth_type::value_type>;

        const auto p = probe<Idx, Args...>(f, probe_type{});

        const auto factory = [dim = p.get_dim()]<typename Acc, typename... _Args>(Acc acc, _Args && ... args)
        {
            return slice<Acc>{dim, acc, std::forward<_Args>(args)...};
        };

        return std::tuple{factory, celerity::access::slice<nth_type::rank>{static_cast<size_t>(p.get_dim())}};
    }

    template <int Rank, size_t... Is>
    auto create_neighbourhood_range_mapper(cl::sycl::range<Rank> range, std::index_sequence<Is...>)
    {
        return celerity::access::neighborhood<Rank>{range[Is]...};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_block_proxy_factory_and_range_mapper(F f)
    {
        using nth_type = nth_t<Idx, Args...>;
        using probe_type = block_probe<typename nth_type::value_type, nth_type::rank>;

        const auto p = probe<Idx, Args...>(f, probe_type{});

        const auto factory = [p]<typename Acc, typename... _Args>(Acc acc, _Args && ... args)
        {
            return block<Acc>(p, acc, std::forward<_Args>(args)...);
        };

        const auto range_mapper = create_neighbourhood_range_mapper(p.size(), std::make_index_sequence<probe_type::rank>{});

        return std::tuple{factory, range_mapper};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_all_proxy_factory_and_range_mapper(F f)
    {
        const auto factory = []<typename Acc, typename... _Args>(Acc acc, _Args && ... args)
        {
            return all<Acc>{acc, std::forward<_Args>(args)...};
        };

        constexpr auto rank = nth_t<Idx, Args...>::rank;
        return std::tuple{factory, celerity::access::all<rank, rank>{}};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_proxy_factory_and_range_mapper(F f)
    {
        using celerity::algorithm::detail::access_type;
        using namespace celerity::access;

        using arg_traits = typename kernel_traits<F, Args...>::template argument<Idx>;

        if constexpr (arg_traits::access_concept == access_type::slice)
        {
            return create_slice_proxy_factory_and_range_mapper<Idx, Args...>(f);
        }
        else if constexpr (arg_traits::access_concept == access_type::chunk)
        {
            return create_block_proxy_factory_and_range_mapper<Idx, Args...>(f);
        }
        else if constexpr (arg_traits::access_concept == access_type::all)
        {
            return create_all_proxy_factory_and_range_mapper<Idx, Args...>(f);
        }
        else if constexpr (arg_traits::access_concept == access_type::one_to_one)
        {
            const auto factory = [](auto acc, auto...) { return acc; };
            return std::tuple{factory, one_to_one<arg_traits::rank>()};
        }
        else
        {
            static_assert(std::is_void_v<F>, "unrecognized access type");
        }
    }
} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_PROBING_H