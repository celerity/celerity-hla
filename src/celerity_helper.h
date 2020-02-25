#ifndef CELERITY_HELPER_H
#define CELERITY_HELPER_H
#define CELERITY_STRICT_CGF_SAFETY 0
#include <celerity.h>

namespace celerity
{
    template <typename ElementTypeA, int RankA,
              typename ElementTypeB, int RankB,
              std::enable_if_t<!std::is_same_v<ElementTypeA, ElementTypeB> ||
                             RankA != RankB, int> = 0>
    bool are_equal(buffer<ElementTypeA, RankA> a, buffer<ElementTypeB, RankB> b)
    {
        return false;
    }

    template <typename ElementType, int Rank>
    bool are_equal(buffer<ElementType, Rank> a, buffer<ElementType, Rank> b)
    {
        return a.get_id() == b.get_id();
    }
}



#endif