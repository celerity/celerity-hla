#ifndef CELERITY_HLA_SLICE_H
#define CELERITY_HLA_SLICE_H

#include "../celerity_helper.h"
#include "../celerity_accessor_traits.h"
#include "concepts.h"
#include "inactive_probe.h"

namespace algo = celerity::hla;

namespace celerity::hla::experimental
{
    template <typename T>
    class slice_probe
    {
    public:
        using value_type = T;

        slice_probe() = default;

        void configure(int dim)
        {
            //assert(dim_ >= 0 && "invalid dim");
            dim_ = dim;

            throw *this;
        }

        int get_dim() const
        {
            //assert(dim_ >= 0 && "dim not set");
            return dim_;
        }

        int get_range() const { return {}; }

        value_type operator*() const
        {
            return {};
        }

        value_type operator[](int) const
        {
            return {};
        }

        slice_probe(const slice_probe &) = default;
        slice_probe(slice_probe &&) = default;
        slice_probe &operator=(const slice_probe &) = delete;
        slice_probe &operator=(slice_probe &&) = delete;

        template <typename V>
        slice_probe &operator=(const V &) = delete;

    private:
        int dim_ = -1;
    };

    template <typename Acc>
    class slice
    {
    public:
        static_assert(algo::traits::is_device_accessor_v<Acc>);

        using accessor_type = Acc;
        using value_type = algo::traits::accessor_value_type_t<Acc>;
        static constexpr auto rank = algo::traits::accessor_rank_v<Acc>;

        slice(const int dim, Acc acc, const cl::sycl::item<rank> item, const cl::sycl::range<rank> range)
            : dim_(dim), range_(range[dim_]), item_(std::move(item)), acc_(std::move(acc))
        {
        }

        void configure(int) {}

        int get_range() const { return range_; }

        value_type operator*() const
        {
            return acc_[item_];
        }

        auto operator[](int pos) const
        {
            auto id = item_.get_id();
            id[dim_] = pos;
            return acc_[id];
        }

        slice(const slice<Acc> &) = default;
        slice(slice<Acc> &&) = default;
        slice<Acc> &operator=(const slice<Acc> &) = delete;
        slice<Acc> &operator=(slice<Acc> &&) = delete;

        template <typename V>
        slice<Acc> &operator=(const V &) = delete;

    private:
        const int dim_;
        const int range_;
        const cl::sycl::item<rank> item_;
        accessor_type acc_;
    };

    template <typename T>
    struct is_slice : std::bool_constant<false>
    {
    };

    template <typename T>
    struct is_slice<slice<T>> : std::bool_constant<true>
    {
    };

    template <typename T>
    struct is_slice<slice_probe<T>> : std::bool_constant<true>
    {
    };

    template <typename T>
    concept AnySlice = celerity::hla::experimental::is_slice<T>::value || InactiveProbe<T>;

    template <typename T>
    concept StrictSlice = is_slice<T>::value;

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_SLICE_H