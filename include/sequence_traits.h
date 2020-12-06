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
struct size {};

template <typename T>
struct size<T, true> : std::integral_constant<int, T::num_actions>
{
};

template <typename T>
constexpr inline auto size_v = size<std::decay_t<T>, is_sequence_v<std::decay_t<T>>>::value;

template <typename T>
struct last_element;

template <typename T>
using last_element_t = typename last_element<std::decay_t<T>>::type;

template <typename T>
struct first_element;

template <typename T>
using first_element_t = typename first_element<std::decay_t<T>>::type;

template <size_t Idx, typename T>
struct nth_element;

template <size_t Idx, typename T>
using nth_element_t = typename nth_element<Idx, std::decay_t<T>>::type;

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