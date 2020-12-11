#ifndef CELERITY_HLA_ACCESSOR_TRAITS_H
#define CELERITY_HLA_ACCESSOR_TRAITS_H

#include "slice.h"
#include "block.h"

#include "../accessor_type.h"

namespace celerity::hla::experimental::traits
{
    template <StrictSlice T>
    constexpr auto get_access_type() { return celerity::algorithm::detail::access_type::slice; }

    template <StrictBlock T>
    constexpr auto get_access_type() { return celerity::algorithm::detail::access_type::chunk; }

    // clang-format off
    template <typename T>
        requires(!AnySlice<T> && !AnyBlock<T>)                    
    constexpr auto get_access_type() { return celerity::algorithm::detail::access_type::one_to_one; }
    // clang-format on

    template <typename T>
    struct is_slice : std::bool_constant<get_access_type<T>() == celerity::algorithm::detail::access_type::slice>
    {
    };

    template <typename T>
    inline constexpr auto is_slice_v = is_slice<T>::value;

    template <typename T>
    struct is_block : std::bool_constant<get_access_type<T>() == celerity::algorithm::detail::access_type::chunk>
    {
    };

    template <typename T>
    inline constexpr auto is_block_v = is_block<T>::value;

    // template <typename T>
    // struct is_item : std::false_type
    // {
    // };

    // template <int Rank>
    // struct is_item<cl::sycl::item<Rank>> : public std::true_type
    // {
    // };

    // template <typename T>
    // inline constexpr auto is_item_v = is_item<T>::value;

    // template <typename T>
    // struct is_all : std::false_type
    // {
    // };

    // template <typename T, int Rank>
    // struct is_all<all<T, Rank>> : std::true_type
    // {
    // };

    // template <typename T>
    // inline constexpr auto is_all_v = is_all<T>::value;

} // namespace celerity::hla::experimental::traits

#endif // ACCESSOR_TRAITS_H