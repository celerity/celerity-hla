#ifndef ANY_ACCESSOR
#define ANY_ACCESSOR

#include "sycl.h"
#include "celerity_accessor_traits.h"

namespace celerity::detail
{

template <typename T>
class any_accessor
{
public:
    using storage_type = std::aligned_storage_t<128>;

    template <typename AccessorType,
              std::enable_if_t<std::is_same_v<T, accessor_value_type_t<AccessorType>>, int> = 0>
    explicit any_accessor(const AccessorType &acc) : rank_(accessor_rank_v<AccessorType>),
                                                     mode_(accessor_mode_v<AccessorType>),
                                                     target_(accessor_target_v<AccessorType>)
    {
        static_assert(sizeof(AccessorType) <= sizeof(storage_type));
        new (&storage_) AccessorType(acc);
    }

    template <int Rank>
    T get(cl::sycl::id<Rank> id) const
    {
        return get<Rank>(mode_, target_, id);
    }

private:
    template <typename AccessorType, int Rank>
    T get(cl::sycl::id<Rank> id) const
    {
        const auto &acc = (*reinterpret_cast<const AccessorType *>(&storage_));
        return acc[id];
    }

    template <int Rank, cl::sycl::access::mode Mode>
    T get(cl::sycl::access::target target, cl::sycl::id<Rank> id) const
    {
        using namespace cl::sycl::access;

        switch (target)
        {
        case target::global_buffer:
            return get<device_accessor<T, Rank, Mode, target::global_buffer>>(id);
        case target::host_buffer:
            return get<host_accessor<T, Rank, Mode>>(id);

        default:
            abort();
            return T{};
        }
    }

    template <int Rank>
    T get(cl::sycl::access::mode mode, cl::sycl::access::target target, cl::sycl::id<Rank> id) const
    {
        using namespace cl::sycl::access;

        switch (mode)
        {
        case mode::read:
            return get<Rank, mode::read>(target, id);
        case mode::write:
            return get<Rank, mode::write>(target, id);
        case mode::read_write:
            return get<Rank, mode::read_write>(target, id);

        default:
            abort();
            return T{};
        }
    }

    const int rank_;
    const cl::sycl::access::mode mode_;
    const cl::sycl::access::target target_;
    storage_type storage_;
};

} // namespace celerity::detail

#endif