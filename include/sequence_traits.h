#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

#include "celerity_helper.h"

#include <utility>

namespace celerity::algorithm::traits
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

template <typename T>
struct last_element;

template <typename T>
using last_element_t = typename last_element<T>::type;

template <typename T>
struct first_element;

template <typename T>
using first_element_t = typename first_element<T>::type;

template <typename T>
struct first_result
{
    using type = T;
};

template <typename... Ts>
struct first_result<std::tuple<Ts...>>
{
    using type = std::tuple_element_t<0, std::tuple<Ts...>>;
};

template <typename T>
using first_result_t = typename first_result<T>::type;

} // namespace celerity::algorithm::traits

#endif