#ifndef FUSION_HELPER_H
#define FUSION_HELPER_H

#include "computation_type.h"
#include "packaged_task_traits.h"

namespace celerity::algorithm
{

std::string to_string(access_type type)
{
    switch (type)
    {
    case access_type::one_to_one:
        return "one_to_one";
    case access_type::slice:
        return "slice";
    case access_type::chunk:
        return "chunk";
    case access_type::all:
        return "all";
    case access_type::item:
        return "item";
    default:
        return "none";
    }
}

std::string to_string(computation_type type)
{
    switch (type)
    {
    case computation_type::generate:
        return "generate";
    case computation_type::transform:
        return "transform";
    case computation_type::zip:
        return "zip";
    case computation_type::reduce:
        return "reduce";
    default:
        return "other";
    }
}

template <typename T>
std::string to_string()
{
    return typeid(T).name();
}

template <typename T, std::enable_if_t<detail::is_packaged_task_v<T>, int> = 0>
void to_string(std::stringstream &ss, T task)
{
    using traits = detail::packaged_task_traits<T>;

    ss << "packaged task:\n";
    ss << "  is t-joint          : " << std::boolalpha << detail::is_t_joint_v<T> << "\n";
    ss << "  type                : " << to_string(traits::computation_type) << "\n";
    ss << "  rank                : " << traits::rank << "\n";
    ss << "  access type         : " << to_string(traits::access_type) << "\n";
    ss << "  input value type    : " << to_string<typename traits::input_value_type>() << "\n";
    ss << "  output value type   : " << to_string<typename traits::output_value_type>() << "\n";
    ss << "  input iterator type : " << to_string<typename traits::input_iterator_type>() << "\n";
    ss << "  output iterator type: " << to_string<typename traits::output_iterator_type>() << "\n";

    if constexpr (traits::computation_type == computation_type::zip)
    {
        using ext_traits = detail::extended_packaged_task_traits<T, computation_type::zip>;

        ss << "\n";
        ss << "  second input access type  : " << to_string(ext_traits::second_input_access_type) << "\n";
        ss << "  second input value type   : " << to_string<typename ext_traits::second_input_value_type>() << "\n";
        ss << "  second input iterator type: " << to_string<typename ext_traits::second_input_iterator_type>() << "\n";
    }

    ss << "\n\n";
}

template <typename T, size_t... Is, std::enable_if_t<is_sequence_v<T>, int> = 0>
void to_string(std::stringstream &ss, T seq, std::index_sequence<Is...>)
{
    ((to_string(ss, std::get<Is>(seq.actions()))), ...);
}

template <typename T, std::enable_if_t<detail::is_packaged_task_sequence_v<T>, int> = 0>
std::string to_string(T seq)
{
    std::stringstream ss{};

    to_string(ss, seq, std::make_index_sequence<size_v<T>>{});

    return ss.str();
}

} // namespace celerity::algorithm

#endif // FUSION_HELPER_H