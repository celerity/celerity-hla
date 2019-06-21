#ifndef STATIC_ITERATOR_H
#define STATIC_ITERATOR_H

#include <array>
#include <type_traits>
#include <utility>

#include "celerity.h"

namespace celerity::algorithm::fixed
{

template<int...Components>
struct static_index
{
	static constexpr auto rank = sizeof...(Components);
	static constexpr std::array<int, rank> components = { Components... };
};

template<typename T, int...Components>
struct static_iterator
{
	using value_type = T;
	using index_type = static_index<Components...>;
	static constexpr size_t rank = index_type::rank;
};

template<size_t Id, typename BeginIter, typename EndIter>
struct static_view
{
	static_assert(std::is_same<typename BeginIter::value_type, typename EndIter::value_type>::value, "same value_type");
	static_assert(BeginIter::rank == EndIter::rank, "same rank");

	using begin_iterator_type = BeginIter;
	using end_iterator_type = EndIter;
	using value_type = typename begin_iterator_type::value_type;

	static constexpr size_t id = Id;
	static constexpr size_t rank = BeginIter::rank;

	explicit static_view(buffer<value_type, rank> buf)
		: buffer_(buf) {}

	auto& buffer() { return buffer_; }

	[[nodiscard]] cl::sycl::range<rank> range() const
	{
		return dispatch_range(std::make_index_sequence<rank>{});
	}

private:
	celerity::buffer<value_type, rank> buffer_;

	template<size_t...Is>
	cl::sycl::range<rank> dispatch_range(std::index_sequence<Is...>) const
	{
		using begin_index_type = typename begin_iterator_type::index_type;
		using end_index_type = typename end_iterator_type::index_type;

		return { (std::get<Is>(begin_index_type::components) - std::get<Is>(end_index_type::components))... };
	}

};

template<template <typename, size_t> typename Buffer, typename T, size_t Rank, size_t...Ids>
constexpr auto dispatch_begin(Buffer<T, Rank>, std::index_sequence<Ids...>)
{
	return static_iterator < T, std::get<Ids>(std::array<int, Rank>{})... > ();
}

template<template <typename, size_t> typename Buffer, typename T, size_t Rank>
constexpr auto begin(Buffer<T, Rank> b)
{
	return dispatch_begin(b, std::make_index_sequence<Rank>{});
}

template<template <typename, size_t> typename Buffer, typename T, size_t Rank>
constexpr auto end(Buffer<T, Rank> b)
{
	return dispatch_begin(b, std::make_index_sequence<Rank>{});
}

template<size_t Id, typename Buffer>
constexpr auto make_view(Buffer buffer)
{
	return static_view<Id, decltype(fixed::begin(buffer)), decltype(fixed::end(buffer))>{buffer};
}

template<access_mode mode, size_t Id, typename BeginIter, typename EndIter>
auto create_accessor(handler cgh, static_view<Id, BeginIter, EndIter> view)
{
	return view.buffer().template get_access<mode>(cgh, view.range());
}

}

#endif