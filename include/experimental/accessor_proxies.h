#ifndef CELERITY_HLA_ACCESSOR_PROXIES_H
#define CELERITY_HLA_ACCESSOR_PROXIES_H

#include "slice.h"
#include "block.h"
#include "probing.h"
#include "../transient.h"

namespace celerity::hla::experimental
{
    template <typename ProxyFactoryType>
    class accessor_base
    {
    public:
        using factory_type = ProxyFactoryType;

        template <typename... Args>
        auto construct_proxy(Args &&... args) const
        {
            return std::invoke(factory_, std::forward<Args>(args)...);
        }

    protected:
        accessor_base(ProxyFactoryType factory)
            : factory_(factory) {}

    private:
        factory_type factory_;
    };

    template <typename ProxyFactoryType, celerity::algorithm::detail::access_type ProxyConcept>
    class accessor : public accessor_base<ProxyFactoryType>
    {
    public:
        using base = accessor_base<ProxyFactoryType>;

        explicit accessor(ProxyFactoryType factory)
            : base(std::move(factory)) {}

        decltype(auto) operator[](const auto item) const
        {
            return base::construct_proxy(item);
        }
    };

    template <celerity::algorithm::detail::access_type ProxyConcept>
    inline auto create_accessor(auto factory, auto acc, auto beg, auto end)
    {
        if constexpr (ProxyConcept == celerity::algorithm::detail::access_type::one_to_one)
        {
            return acc;
        }
        else
        {
            const auto range = beg.get_buffer().get_range();

            const auto proxy_factory = [&, range]() {
                if constexpr (ProxyConcept == celerity::algorithm::detail::access_type::slice)
                {
                    return [=](auto item) { return std::invoke(factory, acc, item, range); };
                }
                else if constexpr (ProxyConcept == celerity::algorithm::detail::access_type::chunk)
                {
                    (void)range;
                    return [=](auto item) { return std::invoke(factory, acc, item); };
                }
                else if constexpr (ProxyConcept == celerity::algorithm::detail::access_type::all)
                {
                    return [=](auto item) { return std::invoke(factory, acc, range); };
                }
            }();

            return accessor<decltype(proxy_factory), ProxyConcept>{proxy_factory};
        }
    }

    // template <typename ExecutionPolicy, cl::sycl::access::mode Mode, template <typename, int> typename Iterator, typename T, int Rank>
    // auto create_accessor(celerity::handler &cgh, Iterator<T, Rank> beg, Iterator<T, Rank> end)
    // {
    //     using namespace celerity::algorithm;
    //     // TODO: move accessor creation into proxy
    //     if constexpr (traits::policy_traits<ExecutionPolicy>::is_distributed)
    //     {
    //         if constexpr (traits::get_accessor_type_<AccessorType>() != algorithm::detail::access_type::all)
    //         {
    //             return beg.get_buffer().template get_access<Mode>(cgh, traits::accessor_traits<Rank, AccessorType>::range_mapper());
    //         }
    //         else
    //         {
    //             if (is_subrange(beg, end)) // TODO
    //             {
    //                 return beg.get_buffer().template get_access<Mode>(cgh, celerity::access::fixed<Rank>({*beg, distance(beg, end)}));
    //             }
    //             else
    //             {
    //                 return beg.get_buffer().template get_access<Mode>(cgh, traits::accessor_traits<Rank, AccessorType>::range_mapper());
    //             }
    //         }
    //     }
    //     else
    //     {
    //         static_assert(traits::is_all_v<AccessorType>, "for master node tasks only all<> is supported");
    //         return beg.get_buffer().template get_access<Mode, cl::sycl::access::target::host_buffer>(cgh, traits::accessor_traits<Rank, AccessorType>::range_mapper());
    //     }
    // }

    template <typename ExecutionPolicy, cl::sycl::access::mode Mode, size_t KernelArgumentIdx, typename KernelType, KernelInput InIterator, KernelInput InSentinel>
    auto get_access(celerity::handler &cgh, InIterator beg, InSentinel end, KernelType kernel)
    {
        using arg_traits = typename kernel_traits<KernelType, InIterator>::template argument<KernelArgumentIdx>;

        const auto [factory, mapper] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<KernelArgumentIdx, InIterator>(kernel);
        const auto acc = beg.get_buffer().template get_access<Mode>(cgh, mapper);

        return create_accessor<arg_traits::access_concept>(factory, acc, beg, end);
    }

    template <typename ExecutionPolicy, cl::sycl::access::mode Mode, size_t KernelArgumentIdx, KernelInput Arg, KernelInput... Args, typename KernelType, KernelInput InIterator, KernelInput InSentinel>
    auto get_access(celerity::handler &cgh, InIterator beg, InSentinel end, KernelType kernel)
    {
        using arg_traits = typename kernel_traits<KernelType, Arg, Args...>::template argument<KernelArgumentIdx>;

        const auto [factory, mapper] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<KernelArgumentIdx, Arg, Args...>(kernel);
        const auto acc = beg.get_buffer().template get_access<Mode>(cgh, mapper);

        return create_accessor<arg_traits::access_concept>(factory, acc, beg, end);
    }

    template <typename ExecutionPolicy, cl::sycl::access::mode Mode, template <typename, int> typename Iterator, typename T, int Rank>
    auto get_out_access(celerity::handler &cgh, Iterator<T, Rank> beg, Iterator<T, Rank>)
    {
        return beg.get_buffer().template get_access<Mode>(cgh, celerity::access::one_to_one<Rank>());
    }

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_ACCESSOR_PROXIES_H