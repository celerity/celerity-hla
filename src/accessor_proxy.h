#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity.h"

namespace celerity
{
	enum class access_type
	{
		one_to_one,
		slice,
		chunk,
	};

	namespace detail
	{
		template <typename T>
		struct function_traits
			: public function_traits<decltype(&T::operator())>
		{};

		template <typename ClassType, typename ReturnType, typename... Args>
		struct function_traits<ReturnType(ClassType::*)(Args...) const>
		{
			static constexpr auto arity = sizeof...(Args);

			using return_type = ReturnType;

			template <size_t I>
			struct arg
			{
				using type = typename std::tuple_element<I, std::tuple<Args...>>::type;
			};
		};
	
		template<size_t Rank, typename F, int I>
		auto get_accessor_type()
		{
			if constexpr (std::is_same_v<slice<Rank>, function_traits<F>::arg<I>::type>)
			{
				return celerity::slice
			}
		}
			
		template<typename F, int...Is>
		auto dispatch_accessor_types(std::index_sequence<Is...>)
		{
			return tuple<
		}
	}

	template<size_t Rank, typename F, int I>
	constexpr auto get_accessor_type()
	{
		using arg_type = typename detail::function_traits<F>::arg<I>::type;

		if constexpr (std::is_same_v<slice<Rank>, arg_type>)
		{
			return celerity::algorithm::iterator_type::slice;
		}
		else if constexpr (std::is_same_v<chunk<Rank>, arg_type>)
		{
			return celerity::algorithm::iterator_type::neighbor;
		}
		else
		{
			return celerity::algorithm::iterator_type::one_to_one;
		}
	}

	template<size_t Rank>
	struct slice{};

	template<size_t Rank>
	struct chunk{};

	template<typename T, size_t Rank, typename AccessorType, algorithm::iterator_type Type>
	class accessor_proxy;

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, algorithm::iterator_type::one_to_one>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		T operator[](const item<Rank> item) const { return accessor_[item]; }
		T& operator[](const item<Rank> item) { return accessor_[item]; }

	private:
		AccessorType accessor_;
	};

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, algorithm::iterator_type::slice>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		slice<Rank> operator[](const item<Rank>) const { return {}; }

	private:
		AccessorType accessor_;
	};

	template<template <typename, size_t, algorithm::iterator_type, celerity::access_mode> typename It,
		typename T, size_t Rank, algorithm::iterator_type Type, celerity::access_mode Mode>
		auto get_access(celerity::handler cgh, It<T, Rank, Type, Mode> beg, It<T, Rank, Type, Mode> end)
	{
		assert(&beg.iterator.buffer() == &end.iterator.buffer());
		assert(*beg.iterator <= *end.iterator);

		auto acc = beg.iterator.buffer().get_access<Mode>(cgh, range<1>{ *end.iterator - *beg.iterator });

		return accessor_proxy<T, Rank, decltype(acc), Type>{ acc };
	}
}

#endif // ACCESSOR_PROXY_H

