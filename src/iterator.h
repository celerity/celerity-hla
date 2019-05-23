#ifndef ITERATOR_H
#define ITERATOR_H

#include <array>
#include <type_traits>
#include <utility>

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

  static constexpr size_t id = Id;
  static constexpr size_t rank = BeginIter::rank;
};

template<typename T, size_t Rank>
struct buffer {};

template<template <typename, size_t> typename Buffer, typename T, size_t Rank, size_t...Ids>
constexpr auto dispatch_begin(Buffer<T, Rank>, std::index_sequence<Ids...>)
{
  return static_iterator<T, std::get<Ids>(std::array<int, Rank>{})...>();
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
  return static_view<Id, decltype(begin(buffer)), decltype(end(buffer))>{};
}

#endif