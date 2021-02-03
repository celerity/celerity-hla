#ifndef PACKAGED_TASK_TRAITS_H
#define PACKAGED_TASK_TRAITS_H

#include "computation_type.h"
#include "partially_packaged_task.h"
#include "require.h"

namespace celerity::algorithm::traits
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
		static constexpr auto computation_type = detail::computation_type::none;

		template <typename, typename>
		static constexpr detail::access_type access_type = detail::access_type::invalid;

		using input_value_type = void;
		using input_iterator_type = void;
		using output_value_type = void;
		using output_iterator_type = void;

		static constexpr bool is_experimental = false;
	};

	template <typename T, detail::computation_type Computation>
	struct extended_packaged_task_traits
	{
		using second_input_iterator_type = void;
		static constexpr detail::access_type second_input_access_type = detail::access_type::invalid;
	};

	template <typename T>
	struct extended_packaged_task_traits<T, detail::computation_type::zip>
	{
		using second_input_iterator_type = void;
		static constexpr detail::access_type second_input_access_type = detail::access_type::invalid;
	};

	template <typename T, typename... Sources>
	constexpr detail::access_type get_second_input_access_type()
	{
		if constexpr (packaged_task_traits<T>::computation_type == detail::computation_type::zip)
		{
			return extended_packaged_task_traits<T, detail::computation_type::zip>::template second_input_access_type<Sources...>;
		}
		else
		{
			return detail::access_type::invalid;
		}
	}

	template <typename T, typename... Sources>
	constexpr inline detail::access_type second_input_access_type_v = get_second_input_access_type<T, Sources...>();

	template <typename T>
	constexpr inline bool is_experimental_v = packaged_task_traits<T>::is_experimental;

	template <typename T>
	struct partially_packaged_task_traits
	{
		static constexpr auto requirement = detail::stage_requirement::invalid;
	};

	template <typename F>
	constexpr inline auto stage_requirement_v = partially_packaged_task_traits<F>::requirement;

	template <typename T, size_t... Is>
	constexpr bool dispatch_is_packaged_task_sequence(std::index_sequence<Is...>)
	{
		return ((is_packaged_task_v<std::tuple_element_t<Is, typename T::actions_t>>)&&...);
	}

	template <typename T, require<is_sequence_v<T>> = yes>
	constexpr bool is_packaged_task_sequence()
	{
		constexpr auto size = T::num_actions;
		return dispatch_is_packaged_task_sequence<T>(std::make_index_sequence<size>{}) && size > 0;
	}

	template <typename T, require<!is_sequence_v<T>> = yes>
	constexpr bool is_packaged_task_sequence()
	{
		return false;
	}

	template <typename F>
	constexpr inline bool is_packaged_task_sequence_v = is_packaged_task_sequence<F>();

} // namespace celerity::algorithm::traits

#endif // PACKAGED_TASK_TRAITS_H