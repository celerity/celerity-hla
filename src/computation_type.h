#ifndef COMPUTATION_TYPE_H
#define COMPUTATION_TYPE_H

#include "kernel_traits.h"

namespace celerity::algorithm
{

enum class computation_type
{
    generate,
    transform,
    reduce,
    zip,
    none
};

} // namespace celerity::algorithm

#endif // COMPUTATION_TYPE_H