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
		invalid,
	};

	namespace detail
	{
		template<typename T, size_t Rank>
		using getter_t = std::function<T(cl::sycl::item<Rank>)>;
	}

	template<typename T>
	struct is_slice : public std::false_type {};

	template<typename T>
	inline constexpr auto is_slice_v = is_slice<T>::value;

	template<typename T, size_t Rank>
	class slice
	{
	public:
		slice(cl::sycl::item<Rank> item, const detail::getter_t<T, Rank>& f)
			: item_(item), getter_(f)
		{}

		const cl::sycl::item<Rank>& item() const { return item_; }
		T operator*() const
		{
			return getter_(item_);
		}
		T operator[](cl::sycl::item<Rank> pos) const { return getter_(pos); }

	private:
		cl::sycl::item<Rank> item_;
		const detail::getter_t<T, Rank>& getter_;
	};
	
	template<typename T, size_t Rank>
	struct is_slice<slice<T, Rank>> : public std::true_type {};
	
	template<typename T>
	struct is_chunk : public std::false_type {};

	template<typename T>
	inline constexpr auto is_chunk_v = is_slice<T>::value;

	template<typename T, size_t Rank>
	struct chunk {};

	template<typename T, size_t Rank>
	struct is_chunk<chunk<T, Rank>> : public std::true_type {};

	namespace detail
	{
		template<typename T, typename = std::void_t<>>
		struct has_call_operator : std::false_type {};

		template<typename T>
		struct has_call_operator<T, std::void_t<decltype(&T::operator())>> : std::true_type {};

		template<class T>
		constexpr inline bool has_call_operator_v = has_call_operator<T>::value;

		template <typename T>
		struct function_traits
			: function_traits<decltype(&T::operator())>
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
	
		template<typename F, int I>
		constexpr std::enable_if_t<has_call_operator_v<F>, access_type>  get_accessor_type()
		{
			using arg_type = typename detail::function_traits<F>::arg<I>::type;

			if constexpr (is_slice_v<arg_type>)
			{
				return access_type::slice;
			}
			else if constexpr (is_chunk_v<arg_type>)
			{
				return access_type::chunk;
			}
			else
			{
				return access_type::one_to_one;
			}
		}

		template<typename F, int>
		constexpr std::enable_if_t<!has_call_operator_v<F>, access_type> get_accessor_type()
		{
			return access_type::invalid;
		}
	}

	template<typename T, size_t Rank, typename AccessorType, access_type Type>
	class accessor_proxy;

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::one_to_one>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		T operator[](const cl::sycl::item<Rank> item) const { return accessor_[item]; }
		T& operator[](const cl::sycl::item<Rank> item) { return accessor_[item]; }

	private:
		AccessorType accessor_;
	};

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::slice>
	{
	public:
		explicit accessor_proxy(AccessorType acc) 
			: accessor_(acc), getter_([this](cl::sycl::item<Rank> i) { return accessor_[i]; }) {}

		slice<T, Rank> operator[](const cl::sycl::item<Rank> it) const
		{
			return slice<T, Rank>{ it, getter_ };
		}

	private:
		detail::getter_t<T, Rank> getter_;
		AccessorType accessor_;
	};

	template<typename T, size_t Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, access_type::chunk>
	{
	public:
		explicit accessor_proxy(AccessorType acc) : accessor_(acc) {}

		chunk<T, Rank> operator[](const cl::sycl::item<Rank>) const { return {}; }

	private:
		AccessorType accessor_;
	};

	template<celerity::access_mode Mode, access_type Type, typename T, size_t Rank>
	auto get_access(celerity::handler cgh, iterator<T, Rank> beg, iterator<T, Rank> end)
	{
		assert(&beg.buffer() == &end.buffer());
		assert(*beg <= *end);

		auto acc = beg.buffer().get_access<Mode>(cgh, distance(beg, end));

		return accessor_proxy<T, Rank, decltype(acc), Type>{ acc };
	}
}

#endif // ACCESSOR_PROXY_H

