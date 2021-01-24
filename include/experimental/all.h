#ifndef CELERITY_HLA_ALL_H
#define CELERITY_HLA_ALL_H

#include "../celerity_helper.h"
#include "../celerity_accessor_traits.h"

namespace algo = celerity::algorithm;

namespace celerity::hla::experimental
{
    template <typename T, size_t Rank>
    class all_probe
    {
    public:
        using value_type = T;
        static constexpr auto rank = Rank;

        all_probe() = default;

        auto operator[](cl::sycl::rel_id<rank> rel_id) const -> value_type { return {}; }
        auto get_range() const -> cl::sycl::range<rank> { return {}; }
    };

    template <class Acc>
    class all
    {
    public:
        using accessor_type = Acc;
        using value_type = algo::traits::accessor_value_type_t<Acc>;
        static constexpr auto rank = algo::traits::accessor_rank_v<Acc>;

        template <typename AccessorType>
        all(AccessorType acc, cl::sycl::range<rank> range)
            : range_(range), acc_(acc)
        {
        }

        auto operator[](cl::sycl::id<rank> id) const -> value_type
        {
            return acc_[id];
        }

        auto get_range() const -> cl::sycl::range<rank>
        {
            return range_;
        }

        all(const all<Acc> &) = delete;
        all(all<Acc> &&) = delete;
        all<Acc> &operator=(const all<Acc> &) = delete;
        all<Acc> &operator=(all<Acc> &&) = delete;
        template <typename V>
        all<Acc> &operator=(const V &) = delete;

    private:
        const cl::sycl::range<rank> range_;
        accessor_type acc_;
    };

    template <typename T>
    struct is_all : std::bool_constant<false>
    {
    };

    template <typename Acc>
    struct is_all<all<Acc>> : std::bool_constant<true>
    {
    };

    template <typename T, size_t Rank>
    struct is_all<all_probe<T, Rank>> : std::bool_constant<true>
    {
    };

    template <typename T>
    concept StrictAll = is_all<T>::value;

    template <typename T>
    concept All = StrictAll<T> || InactiveProbe<T>;

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_ALL_H