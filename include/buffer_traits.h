#ifndef BUFFER_TRAITS_H
#define BUFFER_TRAITS_H

#include "accessors.h"

namespace celerity::algorithm::traits
{

template <typename ElementType, int Rank>
struct buffer_traits
{
    using one = ElementType;

    template <int Dim>
    using slice = slice<ElementType, Dim>;

    template <size_t... Extents>
    using chunk = chunk<ElementType, Extents...>;

    using all = all<ElementType, Rank>;

    using item = cl::sycl::item<Rank>;
};

} // namespace celerity::algorithm::traits

#endif