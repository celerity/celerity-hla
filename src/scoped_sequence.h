#ifndef SCOPED_SEQUENCE_H
#define SCOPED_SEQUENCE_H

#include "sequence.h"

template <typename Sequence, typename Invoker>
class scoped_sequence
{
public:
    explicit scoped_sequence(Sequence sequence, Invoker invoker)
        : sequence_(sequence), invoker_(invoker) {}

    void disable() { invoke = false; }

    scoped_sequence(const scoped_sequence &) = delete;
    scoped_sequence(scoped_sequence &&) = delete;

    scoped_sequence operator=(const scoped_sequence &) = delete;
    scoped_sequence operator=(scoped_sequence &&) = delete;

    ~scoped_sequence()
    {
        if (!invoke)
            return;

        sequence_ | invoker_;
    }

    auto get_invoker() const { return invoker_; }
    auto get_sequence() const { return sequence_; }

    //using invoke_result = decltype(std::declval<Sequence>() | std::declval<Invoker>());

    /*operator invoke_result()
    {
        disable();
        return sequence_ | invoker_;
    }*/

private:
    Sequence sequence_;
    Invoker invoker_;
    bool invoke = true;
};

template <typename Sequence, typename Invoker>
scoped_sequence(Sequence s, Invoker i)->scoped_sequence<Sequence, Invoker>;

auto operator|(celerity::distr_queue &&lhs, celerity::distr_queue &&)
{
    return std::move(lhs);
}

template <typename LhsSequence, typename LhsInvoker, typename RhsSequence, typename RhsInvoker>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &&lhs, scoped_sequence<RhsSequence, RhsInvoker> &&rhs)
{
    lhs.disable();
    rhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs.get_sequence(), lhs.get_invoker() | rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename RhsSequence, typename RhsInvoker>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &lhs, scoped_sequence<RhsSequence, RhsInvoker> &rhs)
{
    lhs.disable();
    rhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs.get_sequence(), lhs.get_invoker() | rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename RhsSequence, typename RhsInvoker>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &&lhs, scoped_sequence<RhsSequence, RhsInvoker> &rhs)
{
    lhs.disable();
    rhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs.get_sequence(), lhs.get_invoker() | rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename RhsSequence, typename RhsInvoker>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &lhs, scoped_sequence<RhsSequence, RhsInvoker> &&rhs)
{
    lhs.disable();
    rhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs.get_sequence(), lhs.get_invoker() | rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &&lhs, U &&rhs)
{
    lhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs, lhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U>
auto operator|(scoped_sequence<LhsSequence, LhsInvoker> &lhs, U &rhs)
{
    lhs.disable();
    return scoped_sequence{lhs.get_sequence() | rhs`, lhs.get_invoker()};
}

#endif // SCOPED_SEQUENCE_H