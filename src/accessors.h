#ifndef ACCESSORS_H
#define ACCESSORS_H

#include <cmath>
#include "inplace_function.h"

#include "any_accessor.h"
#include "variant_item.h"

namespace celerity::algorithm
{

template <int Rank, typename AccessorType>
struct accessor_traits;

class one_to_one
{
};

template <int Rank>
struct accessor_traits<Rank, one_to_one>
{
	static auto range_mapper()
	{
		return celerity::access::one_to_one<Rank>();
	}
};

template <typename T>
struct is_slice : std::false_type
{
};

template <typename T>
inline constexpr auto is_slice_v = is_slice<T>::value;

template <typename T, size_t Dim>
class slice
{
public:
	template <int Rank, cl::sycl::access::mode Mode>
	slice(const cl::sycl::item<Rank> item, cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::host_buffer> acc)
		: idx_(static_cast<int>(item.get_id()[Dim])), item_(item), accessor_(acc)
	{
	}

	template <int Rank, cl::sycl::access::mode Mode>
	slice(const cl::sycl::item<Rank> item, cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc)
		: idx_(static_cast<int>(item.get_id()[Dim])), item_(item), accessor_(acc)
	{
	}

	int index() const { return idx_; }

	T operator*() const
	{
		return this->operator[](idx_);
	}

	T operator[](int pos) const
	{
		item_.apply([pos, acc = accessor_](const auto &item) {
			auto id = item.get_id();
			id[Dim] = pos;
			return acc.template get(id);
		});
	}

	slice<T, Dim> &operator=(const T &)
	{
		assert(false && "cannot assign slice");
		return *this;
	}

private:
	const int idx_;
	const variant_item<2, 3> item_;
	const any_accessor<T> accessor_;
};

template <int Rank, typename T, size_t Dim>
struct accessor_traits<Rank, slice<T, Dim>>
{
	static auto range_mapper()
	{
		return celerity::access::slice<Rank>(Dim);
	}
};

template <typename T, size_t Dim>
struct is_slice<slice<T, Dim>> : std::true_type
{
};

template <typename T>
struct is_chunk : std::false_type
{
};

template <typename T>
inline constexpr auto is_chunk_v = is_slice<T>::value;

template <typename T, size_t... Extents>
class chunk
{
public:
	static constexpr auto rank = sizeof...(Extents);
	static constexpr std::array<size_t, rank> extents = {Extents...};

	template <typename AccessorType>
	chunk(cl::sycl::item<rank> item, AccessorType acc)
		: item_(item), accessor_(acc)
	{
	}

	auto item() const { return item_; }

	T operator*() const
	{
		return this->operator[]({});
	}

	T operator[](cl::sycl::rel_id<rank> rel_id) const
	{
		auto id = item_.get_id();

		for (auto i = 0u; i < rank; ++i)
			id[i] = static_cast<size_t>(static_cast<long>(id[i]) + rel_id[i]);

		return accessor_.template get(id);
	}

	chunk<T, Extents...> &operator=(const T &)
	{
		assert(false && "cannot assign chunk");
		return *this;
	}

	bool is_on_boundary(cl::sycl::range<rank> range)
	{
		return dispatch_is_on_boundary(range, std::make_index_sequence<rank>());
	}

private:
	const cl::sycl::item<rank> item_;
	const any_accessor<T> accessor_;

	template <size_t... Is>
	bool dispatch_is_on_boundary(cl::sycl::range<rank> range, std::index_sequence<Is...>)
	{
		const auto id = item_.get_id();
		return ((id[Is] < (std::get<Is>(extents) / 2)) || ...) ||
			   ((id[Is] > range[Is] - (std::get<Is>(extents) / 2) - 1) || ...);
	}
};

template <int Rank, typename T, size_t... Extents>
struct accessor_traits<Rank, chunk<T, Extents...>>
{
	static auto range_mapper()
	{
		return celerity::access::neighborhood<Rank>(Extents...);
	}
};

template <typename T, size_t... Extents>
struct is_chunk<chunk<T, Extents...>> : public std::true_type
{
};

template <typename T>
struct is_item : std::false_type
{
};

template <typename T>
inline constexpr auto is_item_v = is_item<T>::value;

template <int Rank>
struct is_item<cl::sycl::item<Rank>> : public std::true_type
{
};

template <typename T>
struct is_all : std::false_type
{
};

template <typename T>
inline constexpr auto is_all_v = is_all<T>::value;

template <typename T, int Rank>
class all
{
public:
	template <typename AccessorType>
	all(const cl::sycl::range<Rank> range, AccessorType acc)
		: range_(range), accessor_(acc)
	{
	}

	T operator[](cl::sycl::id<Rank> id) const
	{
		const auto item = cl::sycl::detail::make_item(id, range_);
		return accessor_.template get(item.get_id());
	}

	all<T, Rank> &operator=(const T &)
	{
		assert(false && "cannot assign all");
		return *this;
	}

private:
	const cl::sycl::range<Rank> range_;
	const any_accessor<T> accessor_;
};

template <int Rank, typename T>
struct accessor_traits<Rank, all<T, Rank>>
{
	static auto range_mapper()
	{
		return celerity::access::all<Rank, Rank>();
	}
};

template <typename T, int Rank>
struct is_all<all<T, Rank>> : std::true_type
{
};

} // namespace celerity::algorithm
#endif // ACCESSORS_H