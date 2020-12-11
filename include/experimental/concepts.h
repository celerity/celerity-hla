#ifndef CELERITY_HLA_CONCEPTS
#define CELERITY_HLA_CONCEPTS

#include <type_traits>

namespace celerity::hla
{
    template <class T, class U>
    concept same_as = std::is_same_v<T, U> &&std::is_same_v<U, T>;

    template <typename T, typename R = void, typename... Args>
    concept Callable = std::is_invocable_v<R(Args...)>;

    struct inactive_probe_t
    {
    };

    template <typename T>
    struct concrete_inactive_probe : inactive_probe_t
    {
        using value_type = T;

        void configure(auto) {}

        template <typename U>
        void configure(std::initializer_list<U>) {}

        template <typename... Args>
        void configure(std::tuple<Args...>) {}

        template <typename... Args>
        T operator[](std::tuple<Args...>) { return {}; }

        template <typename U>
        T operator[](std::initializer_list<U>) { return {}; }

        T operator[](auto) { return {}; }
    };

    template <class T>
    concept InactiveProbe = std::is_base_of_v<inactive_probe_t, T>;
} // namespace celerity::hla

#endif