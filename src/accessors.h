#ifndef ACCESSORS_H
#define ACCESSORS_H

#include <cmath>

#include "any_accessor.h"
#include "variant_item.h"

namespace celerity::algorithm
{

struct one_to_one
{
};

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
		return item_.apply([pos, acc = accessor_](const auto &item) {
			auto id = item.get_id();
			id[Dim] = pos;
			return acc.template get(id);
		});
	}

	template <typename V>
	slice<T, Dim> &operator=(const V &)
	{
		static_assert(std::is_void_v<V>, "cannot assign slice");
		return *this;
	}

private:
	const int idx_;
	const variant_item<2, 3> item_;
	const celerity::detail::any_accessor<T> accessor_;
};

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

	template <typename V>
	chunk<T, Extents...> &operator=(const V &)
	{
		static_assert(std::is_void_v<V>, "cannot assign chunk");
		return *this;
	}

	bool is_on_boundary(cl::sycl::range<rank> range)
	{
		return dispatch_is_on_boundary(range, std::make_index_sequence<rank>());
	}

private:
	const cl::sycl::item<rank> item_;
	const celerity::detail::any_accessor<T> accessor_;

	template <size_t... Is>
	bool dispatch_is_on_boundary(cl::sycl::range<rank> range, std::index_sequence<Is...>)
	{
		const auto id = item_.get_id();
		return ((id[Is] < (std::get<Is>(extents) / 2)) || ...) ||
			   ((id[Is] > range[Is] - (std::get<Is>(extents) / 2) - 1) || ...);
	}
};

template <typename T, int Rank>
class all
{
public:
	template <typename AccessorType>
	all(AccessorType acc)
		: accessor_(acc)
	{
	}

	T operator[](cl::sycl::id<Rank> id) const
	{
		return accessor_.template get(id);
	}

	template <typename V>
	all<T, Rank> &operator=(const V &)
	{
		static_assert(std::is_void_v<V>, "cannot assign all");
		return *this;
	}

private:
	const celerity::detail::any_accessor<T> accessor_;
};

template <int Dim>
using slice_f = slice<float, Dim>;

template <int Dim>
using slice_d = slice<double, Dim>;

template <int Dim>
using slice_i = slice<int, Dim>;

template <int Dim>
using slice_2f = slice<cl::sycl::float2, Dim>;

template <int Dim>
using slice_2d = slice<cl::sycl::double2, Dim>;

template <int Dim>
using slice_2i = slice<cl::sycl::int2, Dim>;

template <int Dim>
using slice_3f = slice<cl::sycl::float3, Dim>;

template <int Dim>
using slice_3d = slice<cl::sycl::double3, Dim>;

template <int Dim>
using slice_3i = slice<cl::sycl::int3, Dim>;

template <size_t... Extents>
using chunk_f = chunk<float, Extents...>;

template <size_t... Extents>
using chunk_d = chunk<double, Extents...>;

template <size_t... Extents>
using chunk_i = chunk<int, Extents...>;

template <size_t... Extents>
using chunk_2f = chunk<cl::sycl::float2, Extents...>;

template <size_t... Extents>
using chunk_2d = chunk<cl::sycl::double2, Extents...>;

template <size_t... Extents>
using chunk_2i = chunk<cl::sycl::int2, Extents...>;

template <size_t... Extents>
using chunk_3f = chunk<cl::sycl::float3, Extents...>;

template <size_t... Extents>
using chunk_3d = chunk<cl::sycl::double3, Extents...>;

template <size_t... Extents>
using chunk_3i = chunk<cl::sycl::int3, Extents...>;

} // namespace celerity::algorithm
#endif // ACCESSORS_H