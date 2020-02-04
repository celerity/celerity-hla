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

template <typename T>
struct last_element;

template <typename T>
using last_element_t = typename last_element<T>::type;

template <typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
last_element_t<T> get_last_element(T seq)
{
	return std::get<T::num_actions - 1>(seq.actions());
}

} // namespace celerity::algorithm

#endif