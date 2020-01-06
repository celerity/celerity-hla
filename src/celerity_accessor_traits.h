#ifndef CELERITY_ACCESSOR_TRAITS_H
#define CELERITY_ACCESSOR_TRAITS_H

#include "celerity.h"

#include <type_traits>

namespace celerity::detail
{

template <typename T, int Rank, cl::sycl::access::mode Mode>
using host_accessor = cl::sycl::accessor<T, Rank, Mode, cl::sycl::access::target::host_buffer>;

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
using device_accessor = cl::sycl::accessor<T, Rank, Mode, Target, cl::sycl::access::placeholder::true_t>;

template <class T>
struct is_host_accessor;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct is_host_accessor<host_accessor<T, Rank, Mode>>
    : std::bool_constant<true>
{
};

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
struct is_host_accessor<device_accessor<T, Rank, Mode, Target>>
    : std::bool_constant<false>
{
};

template <class T>
inline constexpr auto is_host_accessor_v = is_host_accessor<T>::value;

template <class T>
inline constexpr auto is_device_accessor_v = !is_host_accessor<T>::value;

template <class T>
struct accessor_target;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct accessor_target<host_accessor<T, Rank, Mode>>
    : std::integral_constant<cl::sycl::access::target,
                             cl::sycl::access::target::host_buffer>
{
};

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
struct accessor_target<device_accessor<T, Rank, Mode, Target>>
    : std::integral_constant<cl::sycl::access::target, Target>
{
};

template <class T>
inline constexpr auto accessor_target_v = accessor_target<T>::value;

template <class T>
struct accessor_mode;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct accessor_mode<host_accessor<T, Rank, Mode>>
    : std::integral_constant<cl::sycl::access::mode, Mode>
{
};

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
struct accessor_mode<device_accessor<T, Rank, Mode, Target>>
    : std::integral_constant<cl::sycl::access::mode, Mode>
{
};

template <class T>
inline constexpr auto accessor_mode_v = accessor_mode<T>::value;

template <class T>
struct accessor_rank;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct accessor_rank<host_accessor<T, Rank, Mode>>
    : std::integral_constant<int, Rank>
{
};

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
struct accessor_rank<device_accessor<T, Rank, Mode, Target>>
    : std::integral_constant<int, Rank>
{
};

template <class T>
inline constexpr auto accessor_rank_v = accessor_rank<T>::value;

template <class T>
struct accessor_value_type;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct accessor_value_type<host_accessor<T, Rank, Mode>>
{
    using type = T;
};

template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
struct accessor_value_type<device_accessor<T, Rank, Mode, Target>>
{
    using type = T;
};

template <class T>
using accessor_value_type_t = typename accessor_value_type<T>::type;

} // namespace celerity::detail

#endif // CELERITY_ACCESSOR_TRAITS_H