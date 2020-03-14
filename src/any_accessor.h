#ifndef ANY_ACCESSOR
#define ANY_ACCESSOR

#include "sycl.h"
#include "celerity_accessor_traits.h"

namespace celerity::algorithm::detail
{

template <class T>
void destruct(T &o)
{
    o.~T();
}

template <class T>
void placement_copy_construct(void *addr, const T &o)
{
    new (addr) T(o);
}

template <typename T, typename Storage = std::aligned_storage_t<128, 16>>
class any_accessor
{
public:
    using storage_type = Storage;

    template <typename AccessorType,
              std::enable_if_t<std::is_same_v<T, traits::accessor_value_type_t<AccessorType>>, int> = 0>
    explicit any_accessor(const AccessorType &acc) : rank_(traits::accessor_rank_v<AccessorType>),
                                                     mode_(traits::accessor_mode_v<AccessorType>),
                                                     target_(traits::accessor_target_v<AccessorType>)
    {
        static_assert(sizeof(AccessorType) <= sizeof(storage_type));
        placement_copy_construct(&storage_, acc);
    }

    any_accessor(const any_accessor &that) = delete;
    any_accessor(any_accessor &&) = delete;
    any_accessor &operator=(const any_accessor &) = delete;
    any_accessor &operator=(any_accessor &&) = delete;

    ~any_accessor()
    {
        apply([](auto &acc) { destruct(acc); });
    }

    template <int Rank>
    decltype(auto) get(cl::sycl::id<Rank> id) const
    {
        return apply<Rank>(mode_, target_, [=](auto acc) { return acc[id]; });
    }

private:
    template <typename F>
    decltype(auto) apply(F f) const
    {
        return apply(rank_, mode_, target_, f);
    }

    template <typename F>
    decltype(auto) apply(F f)
    {
        return apply(rank_, mode_, target_, f);
    }

    template <typename AccessorType, typename F>
    decltype(auto) apply(F f) const
    {
        const auto &acc = (*reinterpret_cast<const AccessorType *>(&storage_));
        return std::invoke(f, acc);
    }

    template <typename AccessorType, typename F>
    decltype(auto) apply(F f)
    {
        auto &acc = (*reinterpret_cast<AccessorType *>(&storage_));
        return std::invoke(f, acc);
    }

    template <int Rank, cl::sycl::access::mode Mode, typename F>
    decltype(auto) apply(cl::sycl::access::target target, F f) const
    {
        using namespace cl::sycl::access;

        switch (target)
        {
        case target::global_buffer:
            return apply<device_accessor<T, Rank, Mode, target::global_buffer>>(f);
        case target::host_buffer:
            return apply<host_accessor<T, Rank, Mode>>(f);
        default:
            abort();
        }
    }

    template <int Rank, cl::sycl::access::mode Mode, typename F>
    decltype(auto) apply(cl::sycl::access::target target, F f)
    {
        using namespace cl::sycl::access;

        switch (target)
        {
        case target::global_buffer:
            return apply<device_accessor<T, Rank, Mode, target::global_buffer>>(f);
        case target::host_buffer:
            return apply<host_accessor<T, Rank, Mode>>(f);
        default:
            abort();
        }
    }

    template <int Rank, typename F>
    decltype(auto) apply(cl::sycl::access::mode mode, cl::sycl::access::target target, F f) const
    {
        using namespace cl::sycl::access;

        switch (mode)
        {
        case mode::read:
            return apply<Rank, mode::read>(target, f);
        case mode::write:
            return apply<Rank, mode::write>(target, f);
        case mode::read_write:
            return apply<Rank, mode::read_write>(target, f);
        default:
            abort();
        }
    }

    template <int Rank, typename F>
    decltype(auto) apply(cl::sycl::access::mode mode, cl::sycl::access::target target, F f)
    {
        using namespace cl::sycl::access;

        switch (mode)
        {
        case mode::read:
            return apply<Rank, mode::read>(target, f);
        case mode::write:
            return apply<Rank, mode::write>(target, f);
        case mode::read_write:
            return apply<Rank, mode::read_write>(target, f);
        default:
            abort();
        }
    }

    template <typename F>
    decltype(auto) apply(int rank, cl::sycl::access::mode mode, cl::sycl::access::target target, F f) const
    {
        using namespace cl::sycl::access;

        switch (rank)
        {
        case 1:
            return apply<1>(mode, target, f);
        case 2:
            return apply<2>(mode, target, f);
        case 3:
            return apply<3>(mode, target, f);
        default:
            abort();
        }
    }

    template <typename F>
    decltype(auto) apply(int rank, cl::sycl::access::mode mode, cl::sycl::access::target target, F f)
    {
        using namespace cl::sycl::access;

        switch (rank)
        {
        case 1:
            return apply<1>(mode, target, f);
        case 2:
            return apply<2>(mode, target, f);
        case 3:
            return apply<3>(mode, target, f);
        default:
            abort();
        }
    }

    const long rank_;
    const cl::sycl::access::mode mode_;
    const cl::sycl::access::target target_;
    storage_type storage_;
};

} // namespace celerity::algorithm::detail

#endif