#ifndef CELERITY_STD_HELPER_H
#define CELERITY_STD_HELPER_H
#define CELERITY_STRICT_CGF_SAFETY 0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma clang diagnostic ignored "-Wdeprecated-copy"

#include <celerity.h>

#pragma clang diagnostic pop

#include "require.h"

namespace celerity::hla::traits
{
    template <typename T>
    struct is_celerity_buffer : std::false_type
    {
    };

    template <typename T, int Rank>
    struct is_celerity_buffer<buffer<T, Rank>> : std::true_type
    {
    };

    template <typename T>
    constexpr inline bool is_celerity_buffer_v = is_celerity_buffer<T>::value;
} // namespace celerity::hla::traits

namespace celerity::hla::detail
{
    template <typename ElementTypeA, int RankA,
              typename ElementTypeB, int RankB,
              hla::require_one<!std::is_same_v<ElementTypeA, ElementTypeB>, RankA != RankB> = hla::yes>
    bool are_equal(buffer<ElementTypeA, RankA> a, buffer<ElementTypeB, RankB> b)
    {
        return false;
    }

    template <typename ElementType, int Rank>
    bool are_equal(buffer<ElementType, Rank> a, buffer<ElementType, Rank> b)
    {
        return a.get_id() == b.get_id();
    }
} // namespace celerity::hla::detail

#endif