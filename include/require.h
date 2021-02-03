#ifndef REQUIRE_H
#define REQUIRE_H

namespace celerity::hla
{

    template <bool... Requirements>
    using require = std::enable_if_t<((Requirements) && ...), int>;

    template <bool... Requirements>
    using require_one = std::enable_if_t<((Requirements) || ...), int>;

    inline constexpr int yes{};

} // namespace celerity::hla

#endif // REQUIRE_H