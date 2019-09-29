#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity_helper.h"
#include "accessors.h"

#include <type_traits>
#include <cmath>

namespace celerity::algorithm
{
enum class access_type
{
	one_to_one,
	slice,
	chunk,
	item,
	invalid,
};

namespace detail
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

	template <size_t I>
	struct arg
	{
		using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
	};
};

template <typename T, int I, bool has_call_operator>
struct arg_type;

template <typename T, int I>
struct arg_type<T, I, true>
{
	using type = typename function_traits<decltype(&T::operator())>::template arg<I>::type;
};

template <typename T, int I>
using arg_type_t = typename arg_type<T, I, has_call_operator_v<T>>::type;

template <typename ArgType, typename ElementType>
struct accessor_type
{
	using type = std::conditional_t<std::is_same_v<ArgType, ElementType>, one_to_one, ArgType>;
};

template <typename F, int I, typename ElementType>
using accessor_type_t = typename accessor_type<arg_type_t<F, I>, ElementType>::type;

template <typename ArgType>
constexpr access_type get_accessor_type_()
{
	if constexpr (is_slice_v<ArgType>)
	{
		return access_type::slice;
	}
	else if constexpr (is_chunk_v<ArgType>)
	{
		return access_type::chunk;
	}
	else if constexpr (is_item_v<ArgType>)
	{
		return access_type::item;
	}
	else
	{
		return access_type::one_to_one;
	}
}

template <typename F, int I>
constexpr bool in_bounds()
{
	return function_traits<F>::arity < I;
}

template <typename F, int I>
constexpr std::enable_if_t<has_call_operator_v<F>, access_type> get_accessor_type()
{
	return get_accessor_type_<arg_type_t<F, I>>();
}

template <typename F, int>
constexpr std::enable_if_t<!has_call_operator_v<F>, access_type> get_accessor_type()
{
	return access_type::invalid;
}
} // namespace detail

template <typename T, int Rank, typename AccessorType, typename Type>
struct accessor_proxy;

template <typename T, int Rank, typename AccessorType>
struct accessor_proxy<T, Rank, AccessorType, one_to_one>
{
public:
	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank> range) : accessor_(acc) {}

	decltype(auto) operator[](const cl::sycl::item<Rank> item) const { return accessor_[item]; }
	decltype(auto) operator[](const cl::sycl::item<Rank> item) { return accessor_[item]; }

	AccessorType &get_accessor() { return accessor_; }

	AccessorType accessor_;
};

template <typename T, int Rank, typename AccessorType>
struct accessor_proxy<T, Rank, AccessorType, all<T, Rank>>
{
public:
	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank> range)
		: accessor_(acc), range_(range) {}

	all<T, Rank> operator[](const cl::sycl::item<Rank>) const
	{
		return {range_, accessor_};
	}

	AccessorType &get_accessor() { return accessor_; }

private:
	AccessorType accessor_;
	cl::sycl::range<Rank> range_;
};

template <typename T, int Rank, typename AccessorType, size_t Dim>
struct accessor_proxy<T, Rank, AccessorType, slice<T, Dim>>
{
public:
	static_assert(Dim >= 0 && Dim < Rank, "Dim out of bounds");

	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank>)
		: accessor_(acc) {}

	slice<T, Dim> operator[](const cl::sycl::item<Rank> it) const
	{
		return slice<T, Dim>(it, accessor_);
	}

private:
	AccessorType accessor_;
};

template <typename T, int Rank, typename AccessorType, size_t... Extents>
struct accessor_proxy<T, Rank, AccessorType, chunk<T, Extents...>>
{
public:
	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank>) : accessor_(acc) {}

	chunk<T, Extents...> operator[](const cl::sycl::item<Rank> item) const
	{
		return {item, accessor_};
	}

private:
	AccessorType accessor_;
};

template <typename ExecutionPolicy, cl::sycl::access::mode Mode, typename AccessorType, typename T, int Rank>
auto get_access(celerity::handler &cgh, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end)
{
	//assert(&beg.buffer() == &end.buffer());
	//assert(*beg <= *end);

	if constexpr (policy_traits<ExecutionPolicy>::is_distributed)
	{
		auto acc = beg.get_buffer().template get_access<Mode>(cgh, accessor_traits<Rank, AccessorType>::range_mapper());
		return accessor_proxy<T, Rank, decltype(acc), AccessorType>{acc, beg.get_buffer().get_range()};
	}
	else
	{
		static_assert(std::is_same_v<one_to_one, AccessorType>);
		auto acc = beg.get_buffer().template get_access<Mode>(cgh, beg.get_buffer().get_range());
		return accessor_proxy<T, Rank, decltype(acc), AccessorType>{acc, beg.get_buffer().get_range()};
	}
}
} // namespace celerity::algorithm

#endif // ACCESSOR_PROXY_H
