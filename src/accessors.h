#ifndef ACCESSORS_H
#define ACCESSORS_H

#include <cmath>

#include "any_accessor.h"
#include "variant_item.h"

namespace celerity::algorithm
{

struct sycl_marker
{
	cl::sycl::id<0> _;
};

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
		return item_.apply([pos, this](const auto &item) {
			auto id = item.get_id();
			id[Dim] = pos;
			return accessor_.template get(id);
		});
	}

	slice(const slice<T, Dim> &) = delete;
	slice(slice<T, Dim> &&) = delete;
	slice<T, Dim> &operator=(const slice<T, Dim> &) = delete;
	slice<T, Dim> &operator=(slice<T, Dim> &&) = delete;

	template <typename V>
	slice<T, Dim> &operator=(const V &) = delete;

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
		{
			id[i] = static_cast<size_t>(static_cast<long>(id[i]) + rel_id[i]);
		}

		return accessor_.template get(id);
	}

	chunk(const chunk<T, Extents...> &) = delete;
	chunk(chunk<T, Extents...> &&) = delete;
	chunk<T, Extents...> &operator=(const chunk<T, Extents...> &) = delete;
	chunk<T, Extents...> &operator=(chunk<T, Extents...> &&) = delete;

	template <typename V>
	chunk<T, Extents...> &operator=(const V &) = delete;

	T get(cl::sycl::id<rank> abs_id) const
	{
		return accessor_.template get(abs_id);
	}

	bool is_on_boundary() const
	{
		return dispatch_is_on_boundary(item_.get_range(), std::make_index_sequence<rank>());
	}

	bool is_on_boundary(cl::sycl::range<rank> range) const
	{
		return dispatch_is_on_boundary(range, std::make_index_sequence<rank>());
	}

	template <typename F, typename G,
			  std::enable_if_t<std::is_invocable_v<F> && std::is_invocable_v<G>, int> = 0>
	auto discern(F on_bounds_functor, G in_bounds_functor) const
	{
		return is_on_boundary()
				   ? std::invoke(on_bounds_functor)
				   : std::invoke(in_bounds_functor);
	}

	template <typename F, typename G,
			  std::enable_if_t<std::is_invocable_v<G> && std::is_same_v<F, std::invoke_result_t<G>>, int> = 0>
	auto discern(F on_bounds_value, G in_bounds_functor) const
	{
		return is_on_boundary()
				   ? on_bounds_value
				   : std::invoke(in_bounds_functor);
	}

	template <typename F, typename G,
			  std::enable_if_t<std::is_invocable_v<F> && std::is_same_v<G, std::invoke_result_t<F>>, int> = 0>
	auto discern(F on_bounds_functor, G in_bounds_value) const
	{
		return is_on_boundary()
				   ? std::invoke(on_bounds_functor)
				   : in_bounds_value;
	}

private:
	const cl::sycl::item<rank> item_;
	const celerity::detail::any_accessor<T> accessor_;

	template <size_t... Is>
	bool dispatch_is_on_boundary(cl::sycl::range<rank> range, std::index_sequence<Is...>) const
	{
		const auto id = item_.get_id();

		return ((id[Is] < (Extents / 2)) || ...) ||
			   ((static_cast<int>(id[Is]) > static_cast<int>(range[Is]) - static_cast<int>(Extents / 2) - 1) || ...);
	}
};

template <typename ChunkType, int Rank>
class chunk_iterator
{
public:
	template <typename T, size_t... Extents>
	chunk_iterator(const chunk<T, Extents...> &chunk, cl::sycl::id<Rank> center, cl::sycl::id<Rank> pos)
		: offset_((Extents / 2)...), center_(center), it_(pos, {Extents...}), chunk_(chunk)
	{
	}

	bool operator==(const chunk_iterator &rhs)
	{
		return equals(get_id(), rhs.get_id());
	}

	bool operator!=(const chunk_iterator &rhs)
	{
		return !equals(get_id(), rhs.get_id());
	}

	chunk_iterator &operator++()
	{
		++it_;
		return *this;
	}

	[[nodiscard]] auto operator*() const
	{
		const auto id = center_ + *it_ - offset_;

		return chunk_.get(id);
	}

	[[nodiscard]] cl::sycl::id<Rank> get_id() const
	{
		return *it_;
	}

private:
	const cl::sycl::id<Rank> offset_;
	cl::sycl::id<Rank> center_;
	iterator<Rank> it_;
	const ChunkType &chunk_;
};

template <typename AllType, int Rank>
class all_iterator
{
public:
	all_iterator(const AllType &all, cl::sycl::id<Rank> pos, cl::sycl::range<Rank> range)
		: it_(pos, range), all_(all)
	{
	}

	bool operator==(const all_iterator &rhs)
	{
		return equals(get_id(), rhs.get_id());
	}

	bool operator!=(const all_iterator &rhs)
	{
		return !equals(get_id(), rhs.get_id());
	}

	all_iterator &operator++()
	{
		++it_;
		return *this;
	}

	[[nodiscard]] auto operator*() const
	{
		return all_[get_id()];
	}

	[[nodiscard]] cl::sycl::id<Rank> get_id() const
	{
		return *it_;
	}

private:
	iterator<Rank> it_;
	const AllType &all_;
};

template <typename T, int Rank, size_t... Extents>
chunk_iterator(const chunk<T, Extents...> &, cl::sycl::id<Rank>, cl::sycl::id<Rank>)->chunk_iterator<chunk<T, Extents...>, Rank>;

template <typename T, size_t... Extents>
auto begin(const chunk<T, Extents...> &chunk)
{
	return chunk_iterator{chunk, chunk.item().get_id(), cl::sycl::id<sizeof...(Extents)>{}};
}

template <typename T, size_t... Extents>
auto end(const chunk<T, Extents...> &chunk)
{
	return chunk_iterator{chunk, chunk.item().get_id(), cl::sycl::id<sizeof...(Extents)>{Extents...}};
}

template <typename T, int Rank>
class all
{
public:
	template <typename AccessorType>
	all(AccessorType acc, cl::sycl::range<Rank> range)
		: accessor_(acc), range_(range)
	{
	}

	T operator[](cl::sycl::id<Rank> id) const
	{
		return accessor_.template get(id);
	}

	all(const all<T, Rank> &) = delete;
	all(all<T, Rank> &&) = delete;
	all<T, Rank> &operator=(const all<T, Rank> &) = delete;
	all<T, Rank> &operator=(all<T, Rank> &&) = delete;

	template <typename V>
	all<T, Rank> &operator=(const V &) = delete;

	auto get_range() const { return range_; }

private:
	const sycl_marker _ = {};
	const celerity::detail::any_accessor<T> accessor_;
	const cl::sycl::range<Rank> range_;
};

template <typename T, int Rank>
auto begin(const all<T, Rank> &all)
{
	return all_iterator{all, {}, all.get_range()};
}

template <typename T, int Rank>
auto end(const all<T, Rank> &all)
{
	return all_iterator{all, all.get_range(), all.get_range()};
}

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

using all_f = all<float, 1>;

using all_d = all<double, 1>;

using all_i = all<int, 1>;

using all_f2 = all<float, 2>;

using all_d2 = all<double, 2>;

using all_i2 = all<int, 2>;

using all_f3 = all<float, 3>;

using all_d3 = all<double, 3>;

using all_i3 = all<int, 3>;

using all_2f = all<cl::sycl::float2, 1>;

using all_2d = all<cl::sycl::double2, 1>;

using all_2i = all<cl::sycl::int2, 1>;

using all_3f = all<cl::sycl::float3, 1>;

using all_3d = all<cl::sycl::double3, 1>;

using all_3i = all<cl::sycl::int3, 1>;

using all_2f2 = all<cl::sycl::float2, 2>;

using all_2d2 = all<cl::sycl::double2, 2>;

using all_2i2 = all<cl::sycl::int2, 2>;

using all_3f2 = all<cl::sycl::float3, 2>;

using all_3d2 = all<cl::sycl::double3, 2>;

using all_3i2 = all<cl::sycl::int3, 2>;

using all_2f3 = all<cl::sycl::float2, 3>;

using all_2d3 = all<cl::sycl::double2, 3>;

using all_2i3 = all<cl::sycl::int2, 3>;

using all_3f3 = all<cl::sycl::float3, 3>;

using all_3d3 = all<cl::sycl::double3, 3>;

using all_3i3 = all<cl::sycl::int3, 3>;

} // namespace celerity::algorithm
#endif // ACCESSORS_H