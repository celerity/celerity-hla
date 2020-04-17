#ifndef ACCESSOR_TYPE_H
#define ACCESSOR_TYPE_H

#include <type_traits>

namespace celerity::algorithm::detail
{

enum class access_type
{
	one_to_one,
	slice,
	chunk,
	item,
	all,
	invalid,
};

}

namespace celerity::algorithm::traits
{

template <typename T, typename = std::void_t<>>
struct has_call_operator : std::false_type
{
};

template <typename T>
struct has_call_operator<T, std::void_t<decltype(&T::operator())>> : std::true_type
{
};

template <class T>
constexpr inline bool has_call_operator_v = has_call_operator<T>::value;

template <typename T>
struct function_traits
	: function_traits<decltype(&T::operator())>
{
};

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
{
	static constexpr auto arity = sizeof...(Args);

	using return_type = ReturnType;

	template <size_t I, bool OutOfBounds>
	struct arg;

	template <size_t I>
	struct arg<I, true>
	{
		using type = void;
	};

	template <size_t I>
	struct arg<I, false>
	{
		using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
	};

	template <size_t I>
	using arg_t = typename arg < I,
		  I<0 || I >= arity>::type;
};

template <typename T>
struct arity : std::integral_constant<size_t, function_traits<T>::arity>
{
};

template <typename T>
inline constexpr auto arity_v = arity<T>::value;

template <typename T, int I, bool has_call_operator>
struct arg_type;

template <typename T, int I>
struct arg_type<T, I, true>
{
	using type = typename function_traits<decltype(&T::operator())>::template arg_t<I>;
};

template <typename T, int I>
using arg_type_t = typename arg_type<T, I, has_call_operator_v<T>>::type;

template <typename T, bool has_call_operator>
struct result_type;

template <typename T>
struct result_type<T, true>
{
	using type = typename function_traits<decltype(&T::operator())>::return_type;
};

template <typename T>
using result_type_t = typename result_type<T, has_call_operator_v<T>>::type;

template <typename ArgType, typename ElementType>
struct accessor_type
{
	using type = std::conditional_t<std::is_same_v<ArgType, ElementType>, detail::one_to_one, ArgType>;
};

template <typename F, int I, typename ElementType>
using accessor_type_t = typename accessor_type<std::decay_t<arg_type_t<F, I>>, ElementType>::type;

template <typename ArgType>
constexpr detail::access_type get_accessor_type_()
{
	using decayed_type = std::decay_t<ArgType>;

	if constexpr (traits::is_slice_v<decayed_type>)
	{
		return detail::access_type::slice;
	}
	else if constexpr (traits::is_chunk_v<decayed_type>)
	{
		return detail::access_type::chunk;
	}
	else if constexpr (traits::is_item_v<decayed_type>)
	{
		return detail::access_type::item;
	}
	else if constexpr (traits::is_all_v<decayed_type>)
	{
		return detail::access_type::all;
	}
	else
	{
		return detail::access_type::one_to_one;
	}
}

template <typename F, int I>
constexpr bool in_bounds()
{
	return function_traits<F>::arity < I;
}

template <typename F, int I>
constexpr std::enable_if_t<has_call_operator_v<F>, detail::access_type> get_accessor_type()
{
	return get_accessor_type_<arg_type_t<F, I>>();
}

template <typename F, int>
constexpr std::enable_if_t<!has_call_operator_v<F>, detail::access_type> get_accessor_type()
{
	static_assert(std::is_void_v<F>, "invalid functor");
	return detail::access_type::invalid;
}

template <typename F, size_t... Is>
constexpr auto dispatch_kernel_result(std::index_sequence<Is...>) -> std::invoke_result_t<F, arg_type_t<F, Is>...>
{
	return {};
}

template <typename F>
using kernel_result_t = std::conditional_t<has_call_operator_v<F>,
										   result_type_t<F>,
										   void>;
} // namespace celerity::algorithm::traits

#endif