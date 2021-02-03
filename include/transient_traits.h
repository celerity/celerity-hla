#ifndef TRANSIENT_TRAITS_H
#define TRANSIENT_TRAITS_H

namespace celerity::hla::traits
{

    template <typename T>
    struct is_transient : std::false_type
    {
    };

    template <typename T>
    inline constexpr bool is_transient_v = is_transient<T>::value;

} // namespace celerity::hla::traits

#endif // TRANSIENT_TRAITS_H