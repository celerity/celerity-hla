#ifndef ACCESSOR_ITERATOR_H
#define ACCESSOR_ITERATOR_H

#include "accessors.h"
#include "iterator.h"

namespace celerity::algorithm
{

template <typename SliceType>
class slice_iterator
{
public:
    static constexpr auto rank = 1;

    template <typename T, size_t Dim, bool Transposed>
    slice_iterator(const slice<T, Dim, Transposed> &slice, cl::sycl::id<rank> pos,
                    cl::sycl::range<rank> range)
        : it_(pos, range), slice_(slice) {}

    bool operator==(const slice_iterator &rhs)
    {
        return detail::equals(get_id(), rhs.get_id());
    }

    bool operator!=(const slice_iterator &rhs)
    {
        return !detail::equals(get_id(), rhs.get_id());
    }

    slice_iterator &operator++()
    {
        ++it_;
        return *this;
    }

    [[nodiscard]] auto operator*() const { return slice_[(*it_)[0]]; }

    [[nodiscard]] cl::sycl::id<rank> get_id() const { return *it_; }

private:
    iterator<rank> it_;
    const SliceType &slice_;
};

template <typename T, size_t Dim, bool Transposed>
slice_iterator(const slice<T, Dim, Transposed> &, cl::sycl::id<1>,
                cl::sycl::range<1>)
    ->slice_iterator<slice<T, Dim, Transposed>>;

template <typename T, size_t Dim, bool Transposed>
auto begin(const slice<T, Dim, Transposed> &s)
{
    return slice_iterator{s, {}, s.get_range()};
}

template <typename T, size_t Dim, bool Transposed>
auto end(const slice<T, Dim, Transposed> &s)
{
    return slice_iterator{s, s.get_range(), s.get_range()};
}

template <typename ChunkType, int Rank>
class chunk_iterator
{
public:
    template <typename T, size_t... Extents>
    chunk_iterator(const chunk<T, Extents...> &chunk, cl::sycl::id<Rank> center,
                    cl::sycl::id<Rank> pos)
        : offset_((Extents / 2)...), center_(center), it_(pos, {Extents...}),
            chunk_(chunk) {}

    bool operator==(const chunk_iterator &rhs)
    {
        return detail::equals(get_id(), rhs.get_id());
    }

    bool operator!=(const chunk_iterator &rhs)
    {
        return !detail::equals(get_id(), rhs.get_id());
    }

    chunk_iterator &operator++()
    {
        ++it_;
        return *this;
    }

    [[nodiscard]] auto operator*() const
    {
        const auto id = center_ + *it_ - offset_;

        return chunk_.get(id);
    }

    [[nodiscard]] cl::sycl::id<Rank> get_id() const { return *it_; }

private:
    const cl::sycl::id<Rank> offset_;
    cl::sycl::id<Rank> center_;
    iterator<Rank> it_;
    const ChunkType &chunk_;
};

template <typename T, int Rank, size_t... Extents>
chunk_iterator(const chunk<T, Extents...> &, cl::sycl::id<Rank>,
                cl::sycl::id<Rank>)
    ->chunk_iterator<chunk<T, Extents...>, Rank>;

template <typename T, size_t... Extents>
auto begin(const chunk<T, Extents...> &chunk)
{
    return chunk_iterator{chunk, chunk.item().get_id(),
                            cl::sycl::id<sizeof...(Extents)>{}};
}

template <typename T, size_t... Extents>
auto end(const chunk<T, Extents...> &chunk)
{
    return chunk_iterator{chunk, chunk.item().get_id(),
                            cl::sycl::id<sizeof...(Extents)>{Extents...}};
}

template <typename AllType, int Rank>
class all_iterator
{
public:
    all_iterator(const AllType &all, cl::sycl::id<Rank> pos,
                    cl::sycl::range<Rank> range)
        : it_(pos, range), all_(all) {}

    bool operator==(const all_iterator &rhs)
    {
        return detail::equals(get_id(), rhs.get_id());
    }

    bool operator!=(const all_iterator &rhs)
    {
        return !detail::equals(get_id(), rhs.get_id());
    }

    all_iterator &operator++()
    {
        ++it_;
        return *this;
    }

    [[nodiscard]] auto operator*() const { return all_[get_id()]; }

    [[nodiscard]] cl::sycl::id<Rank> get_id() const { return *it_; }

private:
    iterator<Rank> it_;
    const AllType &all_;
};

template <typename T, int Rank>
auto begin(const all<T, Rank> &all)
{
    return all_iterator{all, {}, all.get_range()};
}

template <typename T, int Rank>
auto end(const all<T, Rank> &all)
{
    return all_iterator{all, cl::sycl::id<Rank>{all.get_range()},
                        all.get_range()};
}

} // namespace celerity::algorithm

namespace std
{
template <typename AllType, int Rank>
struct iterator_traits<celerity::algorithm::all_iterator<AllType, Rank>>
{
    using difference_type = long;
    using value_type = typename AllType::value_type;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using iterator_category = std::forward_iterator_tag;
};

template <typename SliceType>
struct iterator_traits<celerity::algorithm::slice_iterator<SliceType>>
{
    using difference_type = long;
    using value_type = typename SliceType::value_type;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using iterator_category = std::forward_iterator_tag;
};

template <typename ChunkType, int Rank>
struct iterator_traits<celerity::algorithm::chunk_iterator<ChunkType, Rank>>
{
    using difference_type = long;
    using value_type = typename ChunkType::value_type;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using iterator_category = std::forward_iterator_tag;
};

} // namespace std

#endif // ACCESSOR_ITERATOR_H