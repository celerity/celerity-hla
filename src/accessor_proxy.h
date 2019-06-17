#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity.h"

namespace celerity::algorithm
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
		constexpr auto get_accessor_type()
		{
			using arg_type = typename detail::function_traits<F>::arg<I>::type;

			if constexpr (std::is_same_v<slice<Rank>, arg_type>)
			{
				return access_type::slice;
			}
			else if constexpr (std::is_same_v<chunk<Rank>, arg_type>)
			{
				return access_type::chunk;
			}
			else
			{
				return access_type::one_to_one;
			}
		}
	}

	template<size_t Rank>
	struct slice{};

	template<size_t Rank>
	struct chunk{};

	template<typename T, size_t Rank, typename AccessorType, access_type Type>
	class accessor_proxy;

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::one_to_one>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		T operator[](const item<Rank> item) const { return accessor_[item]; }
		T& operator[](const item<Rank> item) { return accessor_[item]; }

	private:
		AccessorType accessor_;
	};

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::slice>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		slice<Rank> operator[](const item<Rank>) const { return {}; }

	private:
		AccessorType accessor_;
	};

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::chunk>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		chunk<Rank> operator[](const item<Rank>) const { return {}; }

	private:
		AccessorType accessor_;
	};

	template<celerity::access_mode Mode, access_type Type, typename T, size_t Rank>
		auto get_access(celerity::handler cgh, iterator<T, Rank> beg, iterator<T, Rank> end)
	{
		assert(&beg.buffer() == &end.buffer());
		assert(*beg <= *end);

		auto acc = beg.buffer().get_access<Mode>(cgh, range<1>{ *end - *beg });

		return accessor_proxy<T, Rank, decltype(acc), Type>{ acc };
	}
}

#endif // ACCESSOR_PROXY_H

