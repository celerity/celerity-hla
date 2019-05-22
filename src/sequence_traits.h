#ifndef SEQUENCE_TRAITS_H
#define SEQUENCE_TRAITS_H

template<typename T>
struct sequence_traits
{
	using is_sequence_type = std::integral_constant<bool, false>;
};

template<typename T>
constexpr inline bool is_sequence_v = sequence_traits<T>::is_sequence_type::value;

template <typename F, typename... Args>
struct is_invocable :
	std::is_constructible<
	std::function<void(Args ...)>,
	std::reference_wrapper<typename std::remove_reference<F>::type>
	>
{
};

template <typename F, typename... Args>
constexpr inline bool is_invocable_v = ::is_invocable<F, Args...>::value;

template<typename F>
struct is_kernel : std::integral_constant<bool, ::is_invocable_v<F, handler>> { };

template <typename F, typename... Args>
constexpr inline bool is_kernel_v = is_kernel<F>::value;

#endif