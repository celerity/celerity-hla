#ifndef DEPTH_FIRST_H
#define DEPTH_FIRST_H

#include "require.h"
#include "t_joint.h"
#include "packaged_task_traits.h"
#include "sequence.h"

namespace celerity::algorithm::seq
{

    struct end_t
    {
    };

    struct fuse_t
    {
    };

    constexpr inline end_t end{};

    template <typename T, require<traits::is_partially_packaged_task_v<T>> = yes>
    auto operator|(T lhs, const end_t &)
    {
        using namespace detail;
        return fuse(terminate(link(resolve_subranges(sequence(lhs)))));
    }

    template <typename T, require<traits::is_sequence_v<T>,
                                  traits::is_partially_packaged_task_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const end_t &)
    {
        using namespace detail;
        return fuse(terminate(link(resolve_subranges(lhs))));
    }

    template <typename T, require<traits::is_sequence_v<T>, !traits::is_partially_packaged_task_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const end_t &)
    {
        using namespace detail;
        return lhs;
    }

    template <typename T, require<traits::is_partially_packaged_task_v<T>> = yes>
    auto operator|(T lhs, const fuse_t &)
    {
        using namespace detail;
        return fuse(link(resolve_subranges(sequence(lhs))));
    }

    template <typename T, require<traits::is_sequence_v<T>,
                                  traits::is_partially_packaged_task_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const fuse_t &)
    {
        using namespace detail;
        return fuse(link(resolve_subranges(lhs)));
    }

    template <typename T, require<traits::is_sequence_v<T>, !traits::is_partially_packaged_task_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const fuse_t &)
    {
        using namespace detail;
        return lhs;
    }

} // namespace celerity::algorithm::seq

namespace celerity::algorithm::detail
{

    namespace depth_first_impl
    {
        template <typename T, typename F,
                  require<traits::is_t_joint_v<T>> = yes>
        auto traverse(T t_joint, const F &f)
        {
            return make_partial_t_joint(t_joint.get_task(),
                                        std::invoke(f, t_joint.get_secondary()));
        }

        template <typename T, typename F,
                  require<!traits::is_t_joint_v<T>> = yes>
        auto traverse(T task, F)
        {
            return task;
        }

        template <typename T, typename U, typename F>
        auto traverse(T lhs, U rhs, const F &f)
        {
            return sequence(traverse(lhs, f), traverse(rhs, f));
        }

        template <typename T, typename F, typename... Us>
        auto traverse(const sequence<Us...> &lhs, T rhs, const F &f)
        {
            const auto op = [&f](auto &&a, auto &&b) {
                return traverse(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), f);
            };

            return apply_append(lhs, rhs, op);
        }

        template <typename F, typename... Actions, size_t... Is>
        auto traverse(const sequence<Actions...> &s, const F &f, std::index_sequence<Is...>)
        {
            const auto op = [&f](auto &&a, auto &&b) {
                return traverse(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), f);
            };

            return left_fold(s, op);
        }

    } // namespace depth_first_impl

    template <typename F, typename... Actions>
    auto traverse(const sequence<Actions...> &s, const F &f)
    {
        const auto out = [&]() {
            if constexpr (sizeof...(Actions) == 1)
            {
                return depth_first_impl::traverse(s, f);
            }
            else
            {
                return depth_first_impl::traverse(s, f, std::make_index_sequence<sizeof...(Actions)>{});
            }
        }();

        return out;
    }
} // namespace celerity::algorithm::detail

#endif // !DEPTH_FIRST_H