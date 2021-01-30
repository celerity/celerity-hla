#ifndef CELERITY_HLA_INACTIVE_PROBE_H
#define CELERITY_HLA_INACTIVE_PROBE_H

namespace celerity::hla::experimental
{

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

        T operator*() { return {}; }

        template <typename... Args>
        T discern(Args...) { return {}; }

        operator T() { return {}; }
    };

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_INACTIVE_PROBE_H