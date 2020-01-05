#ifndef UTILS_H
#define UTILS_H

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"

#include "../src/catch/catch.hpp"

#define CELERITY_TEST
#include "../src/algorithm.h"

struct GlobalSetupAndTeardown : Catch::TestEventListenerBase
{
    using TestEventListenerBase::TestEventListenerBase;
    void testRunStarting(const Catch::TestRunInfo &) override { celerity::detail::runtime::enable_test_mode(); }
    void testCaseEnded(const Catch::TestCaseStats &) override
    {
        if (celerity::detail::runtime::is_initialized())
        {
            celerity::detail::runtime::teardown();
        }
    }
};

CATCH_REGISTER_LISTENER(GlobalSetupAndTeardown)

template <int Rank, typename T>
std::vector<T> copy_to_host(celerity::distr_queue &q, celerity::buffer<T, Rank> &src)
{
    std::vector<int> dst(src.get_range().size());
    celerity::algorithm::copy(celerity::algorithm::master_blocking(q), src, dst.data());
    return dst;
}

#endif // UTILS_Hcelerity::