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

        void configure(int dim, bool transpose = false)
        {
            //assert(dim_ >= 0 && "invalid dim");
            dim_ = dim;
            transposed_ = transpose;

            throw *this;
        }

        int get_dim() const
        {
            //assert(dim_ >= 0 && "dim not set");
            return dim_;
        }

        bool get_transposed() const
        {
            //assert(dim_ >= 0 && "dim not set");
            return transposed_;
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
        bool transposed_ = false;
    };

    template <typename Acc>
    class slice
    {
    public:
        static_assert(algo::traits::is_device_accessor_v<Acc>);

        using accessor_type = Acc;
        using value_type = algo::traits::accessor_value_type_t<Acc>;
        static constexpr auto rank = algo::traits::accessor_rank_v<Acc>;

        slice(const int dim, const bool transposed, const Acc &acc, const cl::sycl::item<rank> &item, const cl::sycl::range<rank> &range)
            : dim_(dim), range_(range[dim_]), id_([&](auto id) {
                  if constexpr (rank == 2)
                  {
                      if (transposed)
                      {
                          id[1 - dim_] = id_[dim_];
                      }
                  }

                  return id;
              }(item.get_id())),
              acc_(acc)
        {
        }

        void configure(int, bool) const {}
        void configure(int) const {}

        int get_range() const { return range_; }

        auto operator*() const
        {
            return acc_[id_];
        }

        auto operator[](int pos) const
        {
            if constexpr (rank == 1)
            {
                return acc_[{pos}];
            }
            else if constexpr (rank == 2)
            {
                if (dim_ == 0)
                {
                    return acc_[{static_cast<size_t>(pos), id_[1]}];
                }
                else
                {
                    return acc_[{id_[0], static_cast<size_t>(pos)}];
                }
            }
            else
            {
                if (dim_ == 0)
                {
                    return acc_[{static_cast<size_t>(pos), id_[1], id_[2]}];
                }
                else if (dim_ == 1)
                {
                    return acc_[{id_[0], static_cast<size_t>(pos), id_[2]}];
                }
                else
                {
                    return acc_[{id_[0], id_[1], static_cast<size_t>(pos)}];
                }
            }
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
        const cl::sycl::id<rank> id_;
        const accessor_type acc_;
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
    concept Slice = celerity::hla::experimental::is_slice<T>::value || InactiveProbe<T>;

    template <typename T>
    concept StrictSlice = is_slice<T>::value;

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_SLICE_H