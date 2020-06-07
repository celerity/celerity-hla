#ifndef BUFFER_RANGE_H
#define BUFFER_RANGE_H

#include "iterator.h"
#include <type_traits>

namespace celerity::algorithm::detail
{
    template <typename T, int Rank>
    struct buffer_range
    {
        buffer_range(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end)
            : it_beg(beg), it_end(end) {}

        buffer_iterator<T, Rank> it_beg;
        buffer_iterator<T, Rank> it_end;
    };

    template <typename T, int Rank>
    buffer_range(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end)->buffer_range<T, Rank>;

    template <int Rank, typename T>
    auto begin(const buffer_range<T, Rank> &pb)
    {
        return pb.it_beg;
    }

    template <int Rank, typename T>
    auto end(const buffer_range<T, Rank> &pb)
    {
        return pb.it_end;
    }
} // namespace celerity::algorithm::detail

namespace celerity::algorithm::traits
{
    template <typename T>
    struct is_buffer_range : std::false_type
    {
    };

    template <typename T, int Rank>
    struct is_buffer_range<algorithm::detail::buffer_range<T, Rank>> : std::true_type
    {
    };

    template <typename T>
    constexpr inline bool is_buffer_range_v = is_buffer_range<T>::value;
} // namespace celerity::algorithm::traits

#endif