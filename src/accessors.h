#ifndef ACCESSORS_H
#define ACCESSORS_H

#include "sycl.h"
#include <cmath>

namespace celerity::algorithm
{
	namespace detail
	{
		template<typename T>
		using slice_element_getter_t = std::function<T(int)>;

		template<typename T, size_t...Extents>
		using chunk_element_getter_t = std::function<T(cl::sycl::id<sizeof...(Extents)>)>;
	}

	template<typename AccessorType>
	struct accessor_traits;
		
	class one_to_one
	{

	};

	template<>
	struct accessor_traits<one_to_one>
	{
		static auto range_mapper() { return []() {}; }
	};
	
	template<typename T>
	struct is_slice : public std::false_type {};

	template<typename T>
	inline constexpr auto is_slice_v = is_slice<T>::value;

	template<typename T, size_t Dim>
	class slice
	{
	public:
		using getter_t = detail::slice_element_getter_t<T>;

		slice(const int idx, const getter_t& f)
			: idx_(idx), getter_(f)
		{}

		int index() const { return idx_; }

		T operator*() const
		{
			return getter_(idx_);
		}

		T operator[](int pos) const { return getter_(pos); }

	private:
		int idx_;
		const getter_t& getter_;
	};

	template<typename T, size_t Dim>
	struct accessor_traits<slice<T, Dim>>
	{
		static auto range_mapper() { return [](){}; }
	};
	
	template<typename T, size_t Dim>
	struct is_slice<slice<T, Dim>> : public std::true_type {};

	template<typename T>
	struct is_chunk : public std::false_type {};

	template<typename T>
	inline constexpr auto is_chunk_v = is_slice<T>::value;

	template<typename T, size_t...Extents>
	class chunk
	{
	public:
		static constexpr auto rank = sizeof...(Extents);
		static constexpr std::array<size_t, rank> extents = { Extents... };
		using getter_t = detail::chunk_element_getter_t<T, Extents...>;

		chunk(cl::sycl::item<rank> item, const getter_t& f)
			: item_(item), getter_(f)
		{}

		int item() const { return item_; }

		T operator*() const
		{
			return getter_({});
		}

		T operator[](cl::sycl::id<rank> relative) const
		{
			for (auto i = 0; i < rank; ++i)
				assert(std::labs(relative[i]) <= extents[i]);
			
			return getter_(relative);
		}

	private:
		cl::sycl::item<rank> item_;
		const getter_t& getter_;
	};

	template<typename T, size_t...Extents>
	struct accessor_traits<chunk<T, Extents...>>
	{
		static auto range_mapper() { return []() {}; }
	};

	template<typename T, size_t...Extents>
	struct is_chunk<chunk<T, Extents...>> : public std::true_type {};

	template<typename T>
	struct is_item : public std::false_type {};

	template<typename T>
	inline constexpr auto is_item_v = is_item<T>::value;

	template<size_t Rank>
	struct is_item<cl::sycl::item<Rank>> : public std::true_type {};
	
}
#endif // ACCESSORS_H