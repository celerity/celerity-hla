#ifndef CELERITY_HLA_INACTIVE_PROBE_H
#define CELERITY_HLA_INACTIVE_PROBE_H

namespace celerity::hla::experimental
{
    struct any_t
    {
        template<typename T>
        any_t(T&&) {}
    };

    struct inactive_probe_t
    {
    };

    template <typename T>
    struct concrete_inactive_probe final : inactive_probe_t
    {
        using value_type = T;

        void configure(auto...) const {}

        template <typename U>
        void configure(std::initializer_list<U>) const {}

        template <typename... Args>
        void configure(std::tuple<Args...>) const {}

        template <typename... Args>
        T operator[](std::tuple<Args...>) const { return {}; }

        T operator[](std::initializer_list<any_t>) const { return {}; }

        T operator[](auto) const { return {}; }

        T operator*() const { return {}; }

        template <typename... Args>
        T discern(Args...) const { return {}; }

        operator T() const { return {}; }
    };

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_INACTIVE_PROBE_H