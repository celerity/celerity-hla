#ifndef ANY_ACCESSOR
#define ANY_ACCESSOR

#include "sycl.h"

#include <type_traits>

template <typename T>
class any_accessor
{
public:
    using storage_type = std::aligned_storage_t<128>;

    template <int Rank, cl::sycl::access::mode Mode>
    explicit any_accessor(cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::host_buffer> acc)
        : rank_(Rank), mode_(Mode), target_(cl::sycl::access::target::host_buffer)
    {
        init(acc);
    }

    template <int Rank, cl::sycl::access::mode Mode>
    explicit any_accessor(cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc)
        : rank_(Rank), mode_(Mode), target_(cl::sycl::access::target::global_buffer)
    {
        init(acc);
    }

    template <int Rank>
    T get(cl::sycl::id<Rank> id) const
    {
        //assert(Rank == rank_);
        return get<Rank>(mode_, target_, id);
    }

private:
    template <typename AccessorType>
    void init(AccessorType acc)
    {
        static_assert(sizeof(AccessorType) <= sizeof(storage_type));
        new (&storage_) AccessorType(acc);
    }

    template <int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target,
              std::enable_if_t<Target == cl::sycl::access::target::host_buffer, int> = 0>
    T get(cl::sycl::id<Rank> id) const
    {
        return (*reinterpret_cast<const cl::sycl::accessor<T, Rank, Mode, Target> *>(&storage_))[id];
    }

    template <int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target,
              std::enable_if_t<Target == cl::sycl::access::target::global_buffer, int> = 0>
    T get(cl::sycl::id<Rank> id) const
    {
        return (*reinterpret_cast<const cl::sycl::accessor<T, Rank, Mode, Target, cl::sycl::access::placeholder::true_t> *>(&storage_))[id];
    }

    template <int Rank, cl::sycl::access::mode Mode>
    T get(cl::sycl::access::target target, cl::sycl::id<Rank> id) const
    {
        using namespace cl::sycl::access;

        switch (target)
        {
        case target::global_buffer:
            return get<Rank, Mode, target::global_buffer>(id);
        case target::host_buffer:
            return get<Rank, Mode, target::host_buffer>(id);
            //default:
            //assert(false && "unsupported target");
        }

        return T{};
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
            //default:
            //assert(false && "unsupported mode");
        }

        return T{};
    }

    const int rank_;
    const cl::sycl::access::mode mode_;
    const cl::sycl::access::target target_;
    storage_type storage_;
};

#endif