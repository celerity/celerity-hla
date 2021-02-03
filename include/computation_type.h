#ifndef COMPUTATION_TYPE_H
#define COMPUTATION_TYPE_H

#include "kernel_traits.h"

namespace celerity::hla::detail
{

    enum class computation_type
    {
        generate,
        transform,
        reduce,
        zip,
        none
    };

} // namespace celerity::hla::detail

#endif // COMPUTATION_TYPE_H