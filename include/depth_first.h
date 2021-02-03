#ifndef DEPTH_FIRST_H
#define DEPTH_FIRST_H

#include "require.h"
#include "t_joint.h"
#include "packaged_task_traits.h"
#include "sequence.h"

namespace celerity::hla::seq
{
    enum class step : size_t
    {
        resolve_subrange = 0,
        link = 1,
        terminate = 2,
        fuse = 3,
    };

    constexpr inline std::tuple steps = {
        [](const auto &s) { return resolve_subranges(s); },
        [](const auto &s) { return link(s); },
        [](const auto &s) { return terminate(s); },
        [](const auto &s) { return fuse(s); }};

    template <size_t... Steps>
    struct end_t
    {
    };

    struct fuse_t
    {
    };

    template <size_t... Steps>
    struct apply_step;

    template <int CurrentStep, size_t... Steps>
    struct apply_step<CurrentStep, Steps...>
    {
        template <typename T>
        constexpr static auto apply(const T &s)
        {
            const auto &current_step = std::get<CurrentStep>(steps);
            return apply_step<Steps...>::apply(std::invoke(current_step, s));
        }
    };

    template <>
    struct apply_step<>
    {
        template <typename T>
        constexpr static auto apply(const T &s)
        {
            return s;
        }
    };

    template <typename T, size_t... Is>
    auto apply_steps(const T &s, std::index_sequence<Is...>)
    {
        return apply_step<Is...>::apply(s);
    }

    template <size_t... Is, typename T>
    auto apply_steps(const T &s)
    {
        return apply_step<Is...>::apply(s);
    }

    template <size_t ToRemove, typename NewStep, size_t... Steps>
    struct remove_step;

    template <size_t ToRemove, size_t CurrentStep, size_t... Steps, size_t... NewSteps>
    struct remove_step<ToRemove, std::index_sequence<NewSteps...>, CurrentStep, Steps...>
    {
        using type = std::conditional_t<ToRemove == CurrentStep,
                                        typename remove_step<ToRemove, std::index_sequence<NewSteps...>, Steps...>::type,
                                        typename remove_step<ToRemove, std::index_sequence<NewSteps..., CurrentStep>, Steps...>::type>;
    };

    template <size_t ToRemove, size_t... NewSteps>
    struct remove_step<ToRemove, std::index_sequence<NewSteps...>>
    {
        using type = std::index_sequence<NewSteps...>;
    };

    template <step ToRemove, size_t... Steps>
    using remove_step_t = typename remove_step<static_cast<size_t>(ToRemove), std::index_sequence<>, Steps...>::type;

    template <auto... Is>
    constexpr inline auto dispatch_make_all_steps_end(std::index_sequence<Is...>)
    {
        return end_t<Is...>{};
    }

    constexpr inline auto make_all_steps_end()
    {
        return dispatch_make_all_steps_end(std::make_index_sequence<std::tuple_size_v<decltype(steps)>>{});
    }

    constexpr inline auto end = make_all_steps_end();

    template <typename T, size_t... Steps, require<traits::is_partially_packaged_task_v<T>> = yes>
    auto operator|(T lhs, const end_t<Steps...> &e)
    {
        return apply_steps<Steps...>(sequence(lhs));
    }

    template <typename T, size_t... Steps, require<traits::is_sequence_v<T>, traits::is_partially_packaged_task_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const end_t<Steps...> &e)
    {
        using namespace detail;
        return apply_steps<Steps...>(lhs);
    }

    template <typename T, size_t... Steps, require<traits::is_sequence_v<T>, traits::is_celerity_buffer_v<traits::last_element_t<T>>> = yes>
    auto operator|(T lhs, const end_t<Steps...> &)
    {
        using namespace detail;
        return apply_steps(lhs, remove_step_t<step::terminate, Steps...>{});
    }

    /*template <typename T, require<traits::is_partially_packaged_task_v<T>> = yes>
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
    }*/

} // namespace celerity::hla::seq

namespace celerity::hla::detail
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
} // namespace celerity::hla::detail

#endif // !DEPTH_FIRST_H