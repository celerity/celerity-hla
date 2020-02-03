#ifndef PACKAGED_TASK_TRAITS_H
#define PACKAGED_TASK_TRAITS_H

namespace celerity::algorithm::detail
{

template <typename T>
struct is_packaged_task : std::bool_constant<std::is_void_v<T>>
{
};

template <typename F>
constexpr inline bool is_packaged_task_v = is_packaged_task<F>::value;

template <typename T, size_t... Is>
constexpr bool dispatch_is_packaged_task_sequence(std::index_sequence<Is...>)
{
    return ((is_packaged_task_v<std::tuple_element_t<Is, typename T::actions_t>>)&&...);
}

template <typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
constexpr bool is_packaged_task_sequence()
{
    return dispatch_is_packaged_task_sequence<T>(std::make_index_sequence<T::num_actions>{});
}

template <typename T, std::enable_if_t<!is_sequence_v<T>, int> = 0>
constexpr bool is_packaged_task_sequence()
{
    return false;
}

template <typename F>
constexpr inline bool is_packaged_task_sequence_v = is_packaged_task_sequence<F>();

} // namespace celerity::algorithm::detail

#endif // PACKAGED_TASK_TRAITS_H