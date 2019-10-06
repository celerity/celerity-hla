#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity_helper.h"
#include "accessors.h"
#include "accessor_traits.h"

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
using accessor_type_t = typename accessor_type<std::decay_t<arg_type_t<F, I>>, ElementType>::type;

template <typename ArgType>
constexpr access_type get_accessor_type_()
{
	using decayed_type = std::decay_t<ArgType>;

	if constexpr (detail::is_slice_v<decayed_type>)
	{
		return access_type::slice;
	}
	else if constexpr (detail::is_chunk_v<decayed_type>)
	{
		return access_type::chunk;
	}
	else if constexpr (detail::is_item_v<decayed_type>)
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

template <typename AccessorType>
class accessor_proxy_base
{
protected:
	accessor_proxy_base(AccessorType acc)
		: accessor_(acc) {}

public:
	const auto &get_accessor() const { return accessor_; }

private:
	AccessorType accessor_;
};

template <typename T, int Rank, typename AccessorType, typename Type>
class accessor_proxy;

template <typename T, int Rank, typename AccessorType>
class accessor_proxy<T, Rank, AccessorType, one_to_one>
	: public accessor_proxy_base<AccessorType>
{
public:
	using base = accessor_proxy_base<AccessorType>;

	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank>)
		: base(acc) {}

	decltype(auto) operator[](const cl::sycl::item<Rank> item) const
	{
		return base::get_accessor()[item];
	}

	decltype(auto) operator[](const cl::sycl::item<Rank> item)
	{
		return base::get_accessor()[item];
	}
};

template <typename T, int Rank, typename AccessorType>
class accessor_proxy<T, Rank, AccessorType, all<T, Rank>>
	: public accessor_proxy_base<AccessorType>
{
public:
	using base = accessor_proxy_base<AccessorType>;

	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank> range)
		: base(acc), range_(range) {}

	all<T, Rank> operator[](const cl::sycl::item<Rank>) const
	{
		return {range_, base::get_accessor()};
	}

private:
	cl::sycl::range<Rank> range_;
};

template <typename T, int Rank, typename AccessorType, size_t Dim>
class accessor_proxy<T, Rank, AccessorType, slice<T, Dim>>
	: public accessor_proxy_base<AccessorType>
{
public:
	using base = accessor_proxy_base<AccessorType>;

	static_assert(Dim >= 0 && Dim < Rank, "Dim out of bounds");

	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank>)
		: base(acc) {}

	slice<T, Dim> operator[](const cl::sycl::item<Rank> item) const
	{
		return {item, base::get_accessor()};
	}
};

template <typename T, int Rank, typename AccessorType, size_t... Extents>
class accessor_proxy<T, Rank, AccessorType, chunk<T, Extents...>>
	: public accessor_proxy_base<AccessorType>
{
public:
	using base = accessor_proxy_base<AccessorType>;

	static_assert(sizeof...(Extents) == Rank, "must specify extent for every dimension");

	explicit accessor_proxy(AccessorType acc, cl::sycl::range<Rank>)
		: base(acc)
	{
	}

	chunk<T, Extents...> operator[](const cl::sycl::item<Rank> item) const
	{
		return {item, base::get_accessor()};
	}
};

template <typename ExecutionPolicy, cl::sycl::access::mode Mode, typename AccessorType, typename T, int Rank>
auto get_access(celerity::handler &cgh, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end)
{
	//assert(&beg.buffer() == &end.buffer());
	//assert(*beg <= *end);

	if constexpr (policy_traits<ExecutionPolicy>::is_distributed)
	{
		auto acc = beg.get_buffer().template get_access<Mode>(cgh, detail::accessor_traits<Rank, AccessorType>::range_mapper());
		return accessor_proxy<T, Rank, decltype(acc), AccessorType>{acc, beg.get_buffer().get_range()};
	}
	else
	{
		static_assert(std::is_same_v<one_to_one, AccessorType>, "range mappers not supported for host access");
		auto acc = beg.get_buffer().template get_access<Mode>(cgh, beg.get_buffer().get_range());
		return accessor_proxy<T, Rank, decltype(acc), AccessorType>{acc, beg.get_buffer().get_range()};
	}
}
} // namespace celerity::algorithm

#endif // ACCESSOR_PROXY_H
