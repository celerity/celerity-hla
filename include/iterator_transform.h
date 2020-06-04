#ifndef ITERATOR_TRANSFORM_H
#define ITERATOR_TRANSFORM_H

#include "iterator.h"

#include <type_traits>

namespace celerity::algorithm::detail
{
    template <int Rank>
    constexpr auto identity = [](iterator<Rank> &, iterator<Rank> &) {};

    template <int Rank>
    struct iterator_transform : std::function<void(iterator<Rank> &, iterator<Rank> &)>
    {
        using Base = std::function<void(iterator<Rank> &, iterator<Rank> &)>;
        using Base::Base;
        using Base::operator=;
    };

} // namespace celerity::algorithm::detail

namespace celerity::algorithm::traits
{
    template <typename T>
    struct is_iterator_transform : std::bool_constant<false>
    {
    };

    template <int Rank>
    struct is_iterator_transform<algorithm::detail::iterator_transform<Rank>> : std::bool_constant<true>
    {
    };

    template <typename T>
    constexpr inline bool is_iterator_transform_v = is_iterator_transform<T>::value;
} // namespace celerity::algorithm::traits

#endif