#ifndef CELERITY_STD_PLATFORM_H
#define CELERITY_STD_PLATFORM_H

#include <cstdio>

#ifdef __SYCL_DEVICE_ONLY__
#define CELERITY_STD_COMPILING_FOR_DEVICE
#else
#ifdef CL_SYCL_LANGUAGE_VERSION
#define CELERITY_STD_COMPILING_FOR_DEVICE
#endif
#endif

namespace celerity::algorithm::detail
{
    inline void on_error(const char *section, const char *msg)
    {
        printf("[celerity-standard][%s] ERR: %s\n", section, msg);
    }

} // namespace celerity::algorithm::detail

#endif