#ifndef BUFFER_RANGE_H
#define BUFFER_RANGE_H

#include "iterator.h"

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

#endif