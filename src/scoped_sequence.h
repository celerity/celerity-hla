#ifndef SCOPED_SEQUENCE_H
#define SCOPED_SEQUENCE_H

#include "sequence.h"
#include "kernel_traits.h"

namespace celerity::algorithm
{

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

    using invoke_result_type = std::conditional_t<std::is_invocable_v<Sequence, Invoker &>,
                                                  std::invoke_result_t<Sequence, Invoker &>,
                                                  void>;

    operator invoke_result_type()
    {
        return std::invoke(sequence_, invoker_);
    }

    auto get_invoker() const { return invoker_; }
    auto get_sequence() const { return sequence_; }

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
    return scoped_sequence{lhs.get_sequence() | rhs, lhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U,
          std::enable_if_t<detail::is_task_decorator_sequence<U>(), int> = 0>
auto operator|(U &lhs, scoped_sequence<LhsSequence, LhsInvoker> &rhs)
{
    rhs.disable();
    return scoped_sequence{lhs | rhs.get_sequence(), rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U,
          std::enable_if_t<detail::is_task_decorator_sequence<U>(), int> = 0>
auto operator|(U &&lhs, scoped_sequence<LhsSequence, LhsInvoker> &rhs)
{
    rhs.disable();
    return scoped_sequence{lhs | rhs.get_sequence(), rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U,
          std::enable_if_t<detail::is_task_decorator_v<U> || detail::is_task_decorator_sequence<U>(), int> = 0>
auto operator|(U &lhs, scoped_sequence<LhsSequence, LhsInvoker> &&rhs)
{
    rhs.disable();
    return scoped_sequence{lhs | rhs.get_sequence(), rhs.get_invoker()};
}

template <typename LhsSequence, typename LhsInvoker, typename U,
          std::enable_if_t<detail::is_task_decorator_v<U> || detail::is_task_decorator_sequence<U>(), int> = 0>
auto operator|(U &&lhs, scoped_sequence<LhsSequence, LhsInvoker> &&rhs)
{
    rhs.disable();
    return scoped_sequence{lhs | rhs.get_sequence(), rhs.get_invoker()};
}

} // namespace celerity::algorithm

#endif // SCOPED_SEQUENCE_H