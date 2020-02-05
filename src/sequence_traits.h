#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

#include "celerity_helper.h"

namespace celerity::algorithm
{

template <typename T>
struct sequence_traits : std::integral_constant<bool, false>
{
};

template <typename T>
constexpr inline bool is_sequence_v = sequence_traits<T>::value;

template <typename T, bool Sequence = false>
struct size : std::integral_constant<int, 1>
{
};

template <typename T>
struct size<T, true> : std::integral_constant<int, T::num_actions>
{
};

template <typename T>
constexpr inline auto size_v = size<T, is_sequence_v<T>>::value;
;

template<typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
constexpr auto get_last_element(const T& s)
{
    return std::get<size_v<T> - 1>(s);
}

template <typename T>
struct last_element;

template <typename T>
using last_element_t = typename last_element<T>::type;

} // namespace celerity::algorithm

#endif