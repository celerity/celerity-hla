#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

template<typename T>
struct sequence_traits
{
	using is_sequence_type = std::integral_constant<bool, false>;
};

template<typename T>
constexpr inline bool is_sequence_v = sequence_traits<T>::is_sequence_type::value;

template<typename F>
struct is_kernel : std::integral_constant<bool, std::is_invocable_v<F, handler>> { };

template <typename F, typename... Args>
constexpr inline bool is_kernel_v = is_kernel<F>::value;

template<typename F>
constexpr inline bool is_argless_invokable_v = std::is_invocable_v<F>;

#endif