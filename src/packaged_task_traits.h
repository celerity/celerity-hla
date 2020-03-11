#ifndef PACKAGED_TASK_TRAITS_H
#define PACKAGED_TASK_TRAITS_H

#include "partially_packaged_task.h"
#include "computation_type.h"

namespace celerity::algorithm::detail
{

template <typename T>
struct is_packaged_task : std::false_type
{
};

template <typename F>
constexpr inline bool is_packaged_task_v = is_packaged_task<F>::value;

template <typename T>
struct is_partially_packaged_task : std::false_type
{
};

template <typename F>
constexpr inline bool is_partially_packaged_task_v = is_partially_packaged_task<F>::value;

template <typename T>
struct packaged_task_traits
{
    static constexpr auto rank = 0;
    static constexpr auto computation_type = computation_type::none;
    static constexpr auto access_type = access_type::invalid;

    using input_value_type = void;
    using input_iterator_type = void;
    using output_value_type = void;
    using output_iterator_type = void;
};

template <typename T, computation_type Computation>
struct extended_packaged_task_traits
{
};

template <typename T>
constexpr access_type get_second_input_access_type()
{
    if constexpr (packaged_task_traits<T>::computation_type == computation_type::zip)
    {
        return extended_packaged_task_traits<T, computation_type::zip>::second_input_access_type;
    }
    else
    {
        return access_type::invalid;
    }
}

template <typename T>
constexpr inline auto second_input_access_type_v = get_second_input_access_type<T>();

template <typename T>
struct partially_packaged_task_traits : packaged_task_traits<T>
{
    static constexpr auto requirement = stage_requirement::invalid;
};

template <typename F>
constexpr inline auto stage_requirement_v = partially_packaged_task_traits<F>::requirement;

template <typename T, size_t... Is>
constexpr bool dispatch_is_packaged_task_sequence(std::index_sequence<Is...>)
{
    return ((is_packaged_task_v<std::tuple_element_t<Is, typename T::actions_t>>)&&...);
}

template <typename T, std::enable_if_t<is_sequence_v<T>, int> = 0>
constexpr bool is_packaged_task_sequence()
{
    constexpr auto size = T::num_actions;
    return dispatch_is_packaged_task_sequence<T>(std::make_index_sequence<size>{}) && size > 0;
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