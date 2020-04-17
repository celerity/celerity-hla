#ifndef COMPUTATION_TYPE_H
#define COMPUTATION_TYPE_H

#include "kernel_traits.h"

namespace celerity::algorithm::detail
{

enum class computation_type
{
    generate,
    transform,
    reduce,
    zip,
    none
};

} // namespace celerity::algorithm::detail

#endif // COMPUTATION_TYPE_H