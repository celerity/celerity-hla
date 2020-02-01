#ifndef DECORATOR_TRAITS_H
#define DECORATOR_TRAITS_H

namespace celerity::algorithm::detail
{

template <typename T>
struct is_task_decorator : std::bool_constant<std::is_void_v<T>>
{
};

template <typename F>
constexpr inline bool is_task_decorator_v = is_task_decorator<F>::value;

template <typename T, size_t... Is>
constexpr bool dispatch_is_task_decorator_sequence(std::index_sequence<Is...>)
{
    return ((is_task_decorator_v<std::tuple_element_t<Is, typename T::actions_t>>)&&...);
}

template <typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
constexpr bool is_task_decorator_sequence()
{
    return dispatch_is_task_decorator_sequence<T>(std::make_index_sequence<T::num_actions>{});
}

template <typename T, std::enable_if_t<!is_sequence_v<T>, int> = 0>
constexpr bool is_task_decorator_sequence()
{
    return false;
}

} // namespace celerity::algorithm::detail

#endif // DECORATOR_TRAITS_H