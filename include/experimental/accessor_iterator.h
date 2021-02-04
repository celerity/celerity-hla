#ifndef CELERITY_HLA_ACCESSOR_ITERATOR_H
#define CELERITY_HLA_ACCESSOR_ITERATOR_H

#include "slice.h"
#include "block.h"
#include "all.h"

#include "../iterator.h"

namespace celerity::hla::experimental
{
    template <StrictSlice SliceType>
    class slice_iterator
    {
    public:
        static constexpr auto rank = 1;

        slice_iterator(const SliceType &slice, cl::sycl::id<rank> pos,
                       cl::sycl::range<rank> range)
            : it_(pos, range), slice_(slice) {}

        bool operator==(const slice_iterator &rhs) const
        {
            return celerity::hla::detail::equals(get_id(), rhs.get_id());
        }

        bool operator!=(const slice_iterator &rhs) const
        {
            return !celerity::hla::detail::equals(get_id(), rhs.get_id());
        }

        slice_iterator &operator++()
        {
            ++it_;
            return *this;
        }

        [[nodiscard]] auto operator*() const { return slice_[(*it_)[0]]; }

        [[nodiscard]] cl::sycl::id<rank> get_id() const { return *it_; }

    private:
        celerity::hla::iterator<rank> it_;
        const SliceType &slice_;
    };

    template <StrictSlice SliceType>
    slice_iterator(const SliceType &slice, cl::sycl::id<SliceType::rank> pos,
                   cl::sycl::range<SliceType::rank> range) -> slice_iterator<SliceType>;

    template <StrictSlice SliceType>
    auto begin(const SliceType &s)
    {
        return slice_iterator{s, {}, s.get_range()};
    }

    template <StrictSlice SliceType>
    auto end(const SliceType &s)
    {
        return slice_iterator{s, s.get_range(), s.get_range()};
    }

    template <StrictBlock BlockType>
    class block_iterator
    {
    public:
        static constexpr auto rank = BlockType::rank;

        block_iterator(const BlockType &chunk, cl::sycl::id<rank> center,
                       cl::sycl::id<rank> pos)
            : offset_(chunk.size() / 2), center_(center), it_(pos, {chunk.size()}), // TODO
              block(chunk)
        {
        }

        bool operator==(const block_iterator &rhs) const
        {
            return celerity::hla::detail::equals(get_id(), rhs.get_id());
        }

        bool operator!=(const block_iterator &rhs) const
        {
            return !celerity::hla::detail::equals(get_id(), rhs.get_id());
        }

        block_iterator &operator++()
        {
            ++it_;
            return *this;
        }

        [[nodiscard]] auto operator*() const
        {
            const auto id = center_ + *it_ - offset_;

            return block.get(id);
        }

        [[nodiscard]] cl::sycl::id<rank> get_id() const { return *it_; }

    private:
        const cl::sycl::id<rank> offset_;
        cl::sycl::id<rank> center_;
        celerity::hla::iterator<rank> it_;
        const BlockType &block;
    };

    template <StrictBlock BlockType>
    block_iterator(const BlockType &, cl::sycl::id<BlockType::rank>,
                   cl::sycl::id<BlockType::rank>)
        -> block_iterator<BlockType>;

    auto begin(const StrictBlock auto &chunk)
    {
        return block_iterator{chunk, chunk.item().get_id(), {}};
    }

    auto end(const StrictBlock auto &chunk)
    {
        return block_iterator{chunk, chunk.item().get_id(), chunk.size()};
    }

    template <InactiveProbe ProbeType>
    class inactive_probe_iterator
    {
    public:
        using value_type = typename ProbeType::value_type;
        static constexpr auto rank = 1;

        constexpr bool operator==(const inactive_probe_iterator &rhs) const
        {
            return true;
        }

        constexpr bool operator!=(const inactive_probe_iterator &rhs) const
        {
            return false;
        }

        constexpr inactive_probe_iterator &operator++()
        {
            return *this;
        }

        [[nodiscard]] constexpr value_type operator*() const { return {}; }

        [[nodiscard]] constexpr cl::sycl::id<rank> get_id() const { return {}; }
    };

    template <InactiveProbe ProbeType>
    auto begin(const ProbeType &s)
    {
        return inactive_probe_iterator<ProbeType>{};
    }

    template <InactiveProbe ProbeType>
    auto end(const ProbeType &s)
    {
        return inactive_probe_iterator<ProbeType>{};
    }

    template <StrictAll AllType>
    class all_iterator
    {
    public:
        static constexpr auto rank = AllType::rank;

        all_iterator(const AllType &all, cl::sycl::id<rank> pos,
                     cl::sycl::range<rank> range)
            : it_(pos, range), all_(all) {}

        bool operator==(const all_iterator &rhs) const
        {
            return equals(get_id(), rhs.get_id());
        }

        bool operator!=(const all_iterator &rhs) const
        {
            return !equals(get_id(), rhs.get_id());
        }

        all_iterator &operator++()
        {
            ++it_;
            return *this;
        }

        [[nodiscard]] auto operator*() const { return all_[get_id()]; }

        [[nodiscard]] cl::sycl::id<rank> get_id() const { return *it_; }

    private:
        celerity::hla::iterator<rank> it_;
        const AllType &all_;
    };

    template <typename T, int Rank>
    auto begin(const StrictAll auto &all)
    {
        return all_iterator{all, {}, all.get_range()};
    }

    template <typename T, int Rank>
    auto end(const StrictAll auto &all)
    {
        return all_iterator{all, cl::sycl::id<Rank>{all.get_range()},
                            all.get_range()};
    }

} // namespace celerity::hla::experimental

namespace std
{
    template <celerity::hla::experimental::StrictSlice SliceType>
    struct iterator_traits<celerity::hla::experimental::slice_iterator<SliceType>>
    {
        using difference_type = long;
        using value_type = typename SliceType::value_type;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using iterator_category = std::forward_iterator_tag;
    };

    template <celerity::hla::experimental::StrictBlock BlockType>
    struct iterator_traits<celerity::hla::experimental::block_iterator<BlockType>>
    {
        using difference_type = long;
        using value_type = typename BlockType::value_type;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using iterator_category = std::forward_iterator_tag;
    };

    template <celerity::hla::experimental::StrictAll AllType>
    struct iterator_traits<celerity::hla::experimental::all_iterator<AllType>>
    {
        using difference_type = long;
        using value_type = typename AllType::value_type;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using iterator_category = std::forward_iterator_tag;
    };

    template <celerity::hla::experimental::InactiveProbe InactiveProbe>
    struct iterator_traits<celerity::hla::experimental::inactive_probe_iterator<InactiveProbe>>
    {
        using difference_type = long;
        using value_type = typename InactiveProbe::value_type;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using iterator_category = std::forward_iterator_tag;
    };

} // namespace std

#endif // CELERITY_HLA_ACCESSOR_ITERATOR_H