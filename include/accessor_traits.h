#ifndef ACCESSOR_TRAITS_H
#define ACCESSOR_TRAITS_H

#include "accessors.h"

namespace celerity::hla::traits
{
    template <int Rank, typename AccessorType>
    struct accessor_traits;

    template <int Rank>
    struct accessor_traits<Rank, detail::one_to_one>
    {
        static auto range_mapper()
        {
            return celerity::access::one_to_one<Rank>();
        }
    };

    template <int Rank, typename T, size_t Dim, size_t IndexDim>
    struct accessor_traits<Rank, slice<T, Dim, IndexDim>>
    {
        static auto range_mapper()
        {
            return celerity::access::slice<Rank>(Dim);
        }
    };

    template <int Rank, typename T, size_t... Extents>
    struct accessor_traits<Rank, chunk<T, Extents...>>
    {
        static auto range_mapper()
        {
            return celerity::access::neighborhood<Rank>((Extents / 2)...);
        }
    };

    template <int Rank, typename T>
    struct accessor_traits<Rank, all<T, Rank>>
    {
        static auto range_mapper()
        {
            return celerity::access::all<Rank, Rank>();
        }
    };

    template <typename T>
    struct is_slice : std::false_type
    {
    };

    template <typename T, size_t Dim, bool Transposed>
    struct is_slice<slice<T, Dim, Transposed>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr auto is_slice_v = is_slice<T>::value;

    template <typename T>
    struct is_chunk : std::false_type
    {
    };

    template <typename T, size_t... Extents>
    struct is_chunk<chunk<T, Extents...>> : public std::true_type
    {
    };

    template <typename T>
    inline constexpr auto is_chunk_v = is_chunk<T>::value;

    template <typename T>
    struct is_item : std::false_type
    {
    };

    template <int Rank>
    struct is_item<cl::sycl::item<Rank>> : public std::true_type
    {
    };

    template <typename T>
    inline constexpr auto is_item_v = is_item<T>::value;

    template <typename T>
    struct is_all : std::false_type
    {
    };

    template <typename T, int Rank>
    struct is_all<all<T, Rank>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr auto is_all_v = is_all<T>::value;

} // namespace celerity::hla::traits

#endif // ACCESSOR_TRAITS_H