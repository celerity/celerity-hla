#ifndef CELERITY_HELPER_H
#define CELERITY_HELPER_H
#define CELERITY_STRICT_CGF_SAFETY 0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#pragma clang diagnostic ignored "-Wreturn-type"

#include <celerity.h>

#pragma clang diagnostic pop

#include "require.h"

namespace celerity::algorithm::traits
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
} // namespace celerity::algorithm::traits

namespace celerity::algorithm::detail
{
template <typename ElementTypeA, int RankA,
          typename ElementTypeB, int RankB,
          algorithm::require_one<!std::is_same_v<ElementTypeA, ElementTypeB>, RankA != RankB> = algorithm::yes>
bool are_equal(buffer<ElementTypeA, RankA> a, buffer<ElementTypeB, RankB> b)
{
    return false;
}

template <typename ElementType, int Rank>
bool are_equal(buffer<ElementType, Rank> a, buffer<ElementType, Rank> b)
{
    return a.get_id() == b.get_id();
}
} // namespace celerity::algorithm::detail

#endif