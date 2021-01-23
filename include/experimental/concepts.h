#ifndef CELERITY_HLA_CONCEPTS
#define CELERITY_HLA_CONCEPTS

#include <type_traits>

#include "traits.h"

namespace celerity::hla::experimental
{
    template <class T, class U>
    concept same_as = std::is_same_v<T, U> &&std::is_same_v<U, T>;

    template <typename T, typename... Args>
    concept Callable = std::is_invocable_v<T, Args...>;

    template <typename T, typename Value>
    concept Kernel = is_kernel_v<T, Value>;

    template <class T>
    concept InactiveProbe = std::is_base_of_v<inactive_probe_t, T>;
} // namespace celerity::hla::experimental

#endif