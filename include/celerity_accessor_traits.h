#ifndef CELERITY_ACCESSOR_TRAITS_H
#define CELERITY_ACCESSOR_TRAITS_H

#include "celerity.h"

#include <type_traits>

namespace celerity::algorithm
{

    namespace detail
    {

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        using host_accessor = celerity::accessor<T, Rank, Mode, cl::sycl::access::target::host_buffer>;

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target,
                  std::enable_if_t<Target != cl::sycl::access::target::host_buffer, bool> = true>
        using device_accessor = celerity::accessor<T, Rank, Mode, Target>;

    } // namespace detail

    namespace traits
    {

        template <class T>
        struct is_host_accessor;

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        struct is_host_accessor<detail::host_accessor<T, Rank, Mode>>
            : std::bool_constant<true>
        {
        };

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
        struct is_host_accessor<detail::device_accessor<T, Rank, Mode, Target>>
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
        struct accessor_target<detail::host_accessor<T, Rank, Mode>>
            : std::integral_constant<cl::sycl::access::target,
                                     cl::sycl::access::target::host_buffer>
        {
        };

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
        struct accessor_target<detail::device_accessor<T, Rank, Mode, Target>>
            : std::integral_constant<cl::sycl::access::target, Target>
        {
        };

        template <class T>
        inline constexpr auto accessor_target_v = accessor_target<T>::value;

        template <class T>
        struct accessor_mode;

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        struct accessor_mode<detail::host_accessor<T, Rank, Mode>>
            : std::integral_constant<cl::sycl::access::mode, Mode>
        {
        };

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
        struct accessor_mode<detail::device_accessor<T, Rank, Mode, Target>>
            : std::integral_constant<cl::sycl::access::mode, Mode>
        {
        };

        template <class T>
        inline constexpr auto accessor_mode_v = accessor_mode<T>::value;

        template <class T>
        struct accessor_rank;

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        struct accessor_rank<detail::host_accessor<T, Rank, Mode>>
            : std::integral_constant<int, Rank>
        {
        };

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
        struct accessor_rank<detail::device_accessor<T, Rank, Mode, Target>>
            : std::integral_constant<int, Rank>
        {
        };

        template <class T>
        inline constexpr auto accessor_rank_v = accessor_rank<T>::value;

        template <class T>
        struct accessor_value_type;

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        struct accessor_value_type<detail::host_accessor<T, Rank, Mode>>
        {
            using type = T;
        };

        template <typename T, int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
        struct accessor_value_type<detail::device_accessor<T, Rank, Mode, Target>>
        {
            using type = T;
        };

        template <class T>
        using accessor_value_type_t = typename accessor_value_type<T>::type;

    } // namespace traits
} // namespace celerity::algorithm
#endif // CELERITY_ACCESSOR_TRAITS_H