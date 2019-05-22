#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

template<typename T>
struct sequence_traits
{
	using is_sequence_type = std::integral_constant<bool, false>;
};

template<typename T>
constexpr inline bool is_sequence_v = sequence_traits<T>::is_sequence_type::value;

#endif