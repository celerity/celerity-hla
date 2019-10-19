#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

#include "celerity_helper.h"

namespace celerity::algorithm
{

template <typename T>
struct sequence_traits
{
	using is_sequence_type = std::integral_constant<bool, false>;
};

template <typename T>
constexpr inline bool is_sequence_v = sequence_traits<T>::is_sequence_type::value;

template <typename F>
struct is_kernel : std::integral_constant<bool, std::is_invocable_v<F, handler &>>
{
};

template <typename F, typename... Args>
constexpr inline bool is_kernel_v = is_kernel<F>::value;

template <typename F>
constexpr inline bool is_argless_invokable_v = std::is_invocable_v<F>;

template<typename T>
struct contains_kernel_sequence : std::false_type{};

template<typename T, size_t...Is>
constexpr inline bool tuple_is_kernel_sequence(std::index_sequence<Is...>)
{
	return ((is_kernel_v<std::tuple_element_t<Is, T>>) && ...);
}

template<typename T>
constexpr inline bool tuple_is_kernel_sequence()
{
	return tuple_is_kernel_sequence<T>(std::make_index_sequence<std::tuple_size_v<T>>{});
}

template<typename...Ts>
struct contains_kernel_sequence<std::tuple<Ts...>>
	: std::bool_constant<tuple_is_kernel_sequence<std::tuple<Ts...>>()> {};

template<typename T>
inline constexpr bool contains_kernel_sequence_v = contains_kernel_sequence<T>::value;

} // namespace celerity::algorithm

#endif