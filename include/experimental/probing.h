#ifndef CELERITY_HLA_PROBING_H
#define CELERITY_HLA_PROBING_H

#include "concepts.h"
#include "../accessor_type.h"
#include "../celerity_helper.h"

#include "slice.h"
#include "block.h"
#include "all.h"

#include "traits.h"
#include "../iterator.h"

namespace celerity::hla::experimental
{
    template <typename T>
    concept One = !Slice<T> && !Block<T> && !All<T>;

    template <typename T>
    concept CallableObject = celerity::hla::traits::has_call_operator_v<std::remove_cv_t<T>>;

    template <typename T>
    requires(Slice<T> || Block<T> || All<T>) //
        auto default_val(T)
    {
        return typename T::value_type{};
    }

    template <One T>
    auto default_val(T)
    {
        return T{};
    }

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
    auto create_slice_proxy_factory_and_range_mapper(F f, auto beg, auto)
    {
        using nth_type = nth_t<Idx, Args...>;
        using probe_type = slice_probe<typename nth_type::value_type>;

        const auto p = probe<Idx, Args...>(f, probe_type{});
        const auto range = beg.get_buffer().get_range();

        const auto factory = [dim = p.get_dim(), transposed = p.get_transposed(), range]<typename Acc>(const Acc &acc, const auto& item) {
            return slice<Acc>{dim, transposed, acc, item, range};
        };

        const auto acc_meta_factory = [p, beg](auto f) {
            return std::invoke(f, beg.get_buffer(), celerity::access::slice<nth_type::rank>{static_cast<size_t>(p.get_dim())});
        };

        return std::tuple{factory, acc_meta_factory};
    }

    template <int Rank, size_t... Is>
    auto create_neighbourhood_range_mapper(cl::sycl::range<Rank> range, std::index_sequence<Is...>)
    {
        return celerity::access::neighborhood<Rank>{range[Is]...};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_block_proxy_factory_and_range_mapper(F f, auto beg, auto)
    {
        using nth_type = nth_t<Idx, Args...>;
        using probe_type = block_probe<typename nth_type::value_type, nth_type::rank>;

        const auto p = probe<Idx, Args...>(f, probe_type{});

        const auto factory = [p]<typename Acc>(Acc acc, const auto& item) {
            return block<Acc>(p, acc, item);
        };

        const auto acc_meta_factory = [p, beg](auto f) {
            return std::invoke(f, beg.get_buffer(), create_neighbourhood_range_mapper(p.size(), std::make_index_sequence<probe_type::rank>{}));
        };

        return std::tuple{factory, acc_meta_factory};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_all_proxy_factory_and_range_mapper(F f, auto beg, auto end)
    {
        const auto range = beg.get_buffer().get_range();
        const auto factory = [range]<typename Acc>(Acc acc, const auto&) {
            return all<Acc>{acc, range};
        };

        constexpr auto rank = nth_t<Idx, Args...>::rank;

        const auto acc_meta_factory = [beg, end](auto f) {
            if (hla::detail::is_subrange(beg, end))
            {
                // return std::invoke(f, beg.get_buffer(), celerity::access::fixed<rank>{{*beg, hla::detail::distance(beg, end)}});
                return std::invoke(f, beg.get_buffer(), celerity::access::fixed<rank>{{*beg, {}}});
            }
            else
            {
                return std::invoke(f, beg.get_buffer(), celerity::access::all<rank, rank>{});
            }
        };

        return std::tuple{factory, acc_meta_factory};
    }

    template <size_t Idx, KernelInput... Args, typename F>
    auto create_proxy_factory_and_range_mapper(F f, auto beg, auto end)
    {
        using celerity::hla::detail::access_type;
        using namespace celerity::access;

        using arg_traits = typename kernel_traits<F, Args...>::template argument<Idx>;

        if constexpr (arg_traits::access_concept == access_type::slice)
        {
            return create_slice_proxy_factory_and_range_mapper<Idx, Args...>(f, beg, end);
        }
        else if constexpr (arg_traits::access_concept == access_type::chunk)
        {
            return create_block_proxy_factory_and_range_mapper<Idx, Args...>(f, beg, end);
        }
        else if constexpr (arg_traits::access_concept == access_type::all)
        {
            return create_all_proxy_factory_and_range_mapper<Idx, Args...>(f, beg, end);
        }
        else if constexpr (arg_traits::access_concept == access_type::one_to_one)
        {
            const auto factory = [](auto acc, auto&&...) { return acc; };
            const auto acc_meta_factory = [beg](auto f) {
                return std::invoke(f, beg.get_buffer(), one_to_one<arg_traits::rank>());
            };

            return std::tuple{factory, acc_meta_factory};
        }
        else
        {
            static_assert(std::is_void_v<F>, "unrecognized access type");
        }
    }
} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_PROBING_H