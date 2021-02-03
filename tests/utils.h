#ifndef UTILS_H
#define UTILS_H

#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"

#include "../include/catch/catch.hpp"

#define CELERITY_TEST
#include "../include/algorithm.h"

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
    std::vector<int> dst(src.get_range().size(), 0);
    celerity::hla::copy(celerity::hla::master_blocking(q), src, dst.data());
    return dst;
}

template <auto Cmp>
inline auto is_equal_to = [](const auto &x) { return x == Cmp; };

template <auto Cmp, typename InputIterator>
bool elements_equal_to(InputIterator beg, InputIterator end)
{
    return std::all_of(beg, end, is_equal_to<Cmp>);
}

template <auto Cmp, typename T>
bool elements_equal_to(const T &range)
{
    return elements_equal_to<Cmp>(begin(range), end(range));
}

#endif // UTILS_Hcelerity::