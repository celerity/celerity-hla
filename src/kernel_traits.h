#ifndef KERNEL_TRAITS_H
#define KERNEL_TRAITS_H

#include <type_traits>

namespace celerity::traits
{
	template<typename A, typename B>
	struct is_combinable
		: std::integral_constant<bool, std::is_same<typename A::view_type, typename B::view_type>::value> {};

	template<typename A, typename B>
	constexpr inline bool is_combinable_v = is_combinable<A, B>::value;
}

#endif
