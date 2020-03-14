#ifndef CELERITY_HELPER_H
#define CELERITY_HELPER_H
#define CELERITY_STRICT_CGF_SAFETY 0

#include <celerity.h>
#include "require.h"

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