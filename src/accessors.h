#ifndef ACCESSORS_H
#define ACCESSORS_H

#include "sycl.h"

namespace celerity::algorithm
{
	namespace detail
	{
		template<typename T>
		using slice_element_getter_t = std::function<T(int)>;
	}

	class one_to_one
	{

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

		slice(int idx, const getter_t& f)
			: idx_(idx), getter_(f)
		{}

		const int index() const { return idx_; }

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
	struct is_slice<slice<T, Dim>> : public std::true_type {};

	template<typename T>
	struct is_chunk : public std::false_type {};

	template<typename T>
	inline constexpr auto is_chunk_v = is_slice<T>::value;

	template<typename T, size_t Rank>
	struct chunk {};

	template<typename T, size_t Rank>
	struct is_chunk<chunk<T, Rank>> : public std::true_type {};

	template<typename T>
	struct is_item : public std::false_type {};

	template<typename T>
	inline constexpr auto is_item_v = is_item<T>::value;

	template<size_t Rank>
	struct is_item<cl::sycl::item<Rank>> : public std::true_type {};
	
}
#endif // ACCESSORS_H