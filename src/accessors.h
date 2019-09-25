#ifndef ACCESSORS_H
#define ACCESSORS_H

#include "sycl.h"
#include <cmath>
#include "inplace_function.h"

namespace celerity::algorithm
{
namespace detail
{
template <typename T>
using slice_element_getter_t = stdext::inplace_function<T(int), 128>;

template <typename T, size_t... Extents>
using chunk_element_getter_t = stdext::inplace_function<T(cl::sycl::rel_id<sizeof...(Extents)>), 128>;

template <typename T, int Rank>
using all_element_getter_t = stdext::inplace_function<T(cl::sycl::id<Rank>), 128>;
} // namespace detail

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

template <typename T, int Rank, cl::sycl::access::mode mode, cl::sycl::access::target target,
		  std::enable_if_t<target == cl::sycl::access::target::host_buffer, int> = 0>
T get_value(const void *accessor, cl::sycl::id<Rank> id)
{
	return (*reinterpret_cast<const cl::sycl::accessor<T, Rank, mode, target> *>(accessor))[id];
}

template <typename T, int Rank, cl::sycl::access::mode mode, cl::sycl::access::target target,
		  std::enable_if_t<target == cl::sycl::access::target::global_buffer, int> = 0>
T get_value(const void *accessor, cl::sycl::id<Rank> id)
{
	return (*reinterpret_cast<const cl::sycl::accessor<T, Rank, mode, target, cl::sycl::access::placeholder::true_t> *>(accessor))[id];
}

template <typename T, int Rank, cl::sycl::access::mode mode>
T get_value(const void *accessor, cl::sycl::access::target target, cl::sycl::id<Rank> id)
{
	using namespace cl::sycl::access;

	switch (target)
	{
	case target::global_buffer:
		return get_value<T, Rank, mode, target::global_buffer>(accessor, id);
	case target::host_buffer:
		return get_value<T, Rank, mode, target::host_buffer>(accessor, id);
	}
}

template <typename T, int Rank>
T get_value(const void *accessor, cl::sycl::access::mode mode, cl::sycl::access::target target, cl::sycl::id<Rank> id)
{
	using namespace cl::sycl::access;

	switch (mode)
	{
	case mode::read:
		return get_value<T, Rank, mode::read>(accessor, target, id);
	case mode::write:
		return get_value<T, Rank, mode::write>(accessor, target, id);
	case mode::read_write:
		return get_value<T, Rank, mode::read_write>(accessor, target, id);
	}
}

template <typename T, size_t Dim>
class slice
{
public:
	using item_store_t = std::array<std::byte, sizeof(cl::sycl::item<3>)>;
	using accessor_store_t = std::array<std::byte, 128>;

	template <int Rank, cl::sycl::access::mode Mode>
	slice(const cl::sycl::item<Rank> item, cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::host_buffer> acc)
		: idx_(static_cast<int>(item.get_id()[Dim])), rank_(Rank), mode_(Mode), target_(cl::sycl::access::target::host_buffer)
	{
		init(item, acc);
	}

	template <int Rank, cl::sycl::access::mode Mode>
	slice(const cl::sycl::item<Rank> item, cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> acc)
		: idx_(static_cast<int>(item.get_id()[Dim])), rank_(Rank), mode_(Mode), target_(cl::sycl::access::target::global_buffer)
	{
		init(item, acc);
	}

	int index() const { return idx_; }

	T operator*() const
	{
		return this->operator[](idx_);
	}

	T operator[](int pos) const
	{

		switch (rank_)
		{
		case 2:
		{
			const auto item = *reinterpret_cast<const cl::sycl::item<2> *>(item_.data());
			auto id = item.get_id();
			id[Dim] = pos;
			return get_value<T, 2>(accessor_.data(), mode_, target_, id);
		}

		case 3:
		{
			const auto item = *reinterpret_cast<const cl::sycl::item<3> *>(item_.data());
			auto id = item.get_id();
			id[Dim] = pos;
			return get_value<T, 3>(accessor_.data(), mode_, target_, id);
		}
		}
	}

	slice<T, Dim> &operator=(const T &)
	{
		assert(false && "cannot assign slice");
		return *this;
	}

private:
	int idx_;
	int rank_;
	cl::sycl::access::mode mode_;
	cl::sycl::access::target target_;

	item_store_t item_;
	accessor_store_t accessor_;

	template<typename ItemType, typename AccessorType>
	void init(ItemType item, AccessorType acc)
	{
		static_assert(sizeof(ItemType) <= sizeof(item_store_t));
		static_assert(sizeof(AccessorType) <= sizeof(accessor_store_t));

		new (item_.data()) ItemType(item);
		new (accessor_.data()) AccessorType(acc);
	}
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
	using getter_t = detail::chunk_element_getter_t<T, Extents...>;

	chunk(cl::sycl::item<rank> item, const getter_t &f)
		: item_(item), getter_(f)
	{
	}

	int item() const { return item_; }

	T operator*() const
	{
		return getter_({});
	}

	T operator[](cl::sycl::rel_id<rank> relative) const
	{
		//for (auto i = 0; i < rank; ++i)
		//	assert(cl::sycl::fabs(relative[i]) <= extents[i]);

		return getter_(relative);
	}

	chunk<T, Extents...> &operator=(const T &)
	{
		assert(false && "cannot assign chunk");
		return *this;
	}

private:
	cl::sycl::item<rank> item_;
	const getter_t getter_;
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
	using getter_t = detail::all_element_getter_t<T, Rank>;

	all(const getter_t &f)
		: getter_(f)
	{
	}

	T operator[](cl::sycl::id<Rank> id) const
	{
		return getter_(id);
	}

	all<T, Rank> &operator=(const T &)
	{
		assert(false && "cannot assign all");
		return *this;
	}

private:
	const getter_t getter_;
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