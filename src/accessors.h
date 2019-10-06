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

} // namespace celerity::algorithm
#endif // ACCESSORS_H