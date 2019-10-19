#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <optional>
#include <variant>
#include <assert.h>

#include "sequence_traits.h"

namespace celerity::algorithm
{
struct no_result_t
{
};

template <typename T>
inline constexpr bool is_no_result_v = std::is_same_v<no_result_t, T>;

template <typename T, size_t... Is>
inline constexpr bool is_empty_result_v = (is_no_result_v<std::tuple_element_t<Is, T>> && ...);

template <typename... Actions>
class sequence
{
public:
	using actions_t = std::tuple<Actions...>;
	static constexpr auto num_actions = sizeof...(Actions);

	sequence(Actions... actions)
		: actions_(actions...)
	{
	}

	sequence(actions_t actions)
		: actions_(actions)
	{
	}

	template <typename... SequenceActions, typename Action>
	sequence(sequence<SequenceActions...> &&seq, Action action)
		: sequence(std::move(seq), action, std::index_sequence_for<SequenceActions...>{})
	{
	}

	template <typename... Args>
	decltype(auto) operator()(Args &&... args) const
	{
		if constexpr (num_actions == 1 && std::is_void_v<std::invoke_result_t<std::tuple_element_t<0, actions_t>, Args...>>)
		{
			invoke(std::get<0>(actions_), std::forward<Args>(args)...);
			return;
		}

		return dispatch(std::make_index_sequence<num_actions>{}, std::forward<Args>(args)...);
	}

	constexpr actions_t &actions() { return actions_; }

private:
	actions_t actions_;

	template <typename... SequenceActions, typename Action, size_t... Ids>
	sequence(sequence<SequenceActions...> &&sequence, Action action, std::index_sequence<Ids...>)
		: actions_(std::move(std::get<Ids>(sequence.actions()))..., action)
	{
	}

	template <typename Invocable, typename... Args>
	decltype(auto) invoke(const Invocable &invocable, Args &&... args) const
	{
		if constexpr (std::is_invocable_v<Invocable, Args...>)
		{
			if constexpr (std::is_void_v<std::invoke_result_t<Invocable, Args...>>)
			{
				std::invoke(invocable, std::forward<Args>(args)...);
				return no_result_t{};
			}
			else
			{
				return std::invoke(invocable, std::forward<Args>(args)...);
			}
		}
		else if constexpr (std::is_invocable_v<Invocable>)
		{
			if constexpr (std::is_void_v<std::invoke_result_t<Invocable>>)
			{
				std::invoke(invocable);
				return no_result_t{};
			}
			else
			{
				return std::invoke(invocable);
			}
		}
		else
		{
			assert((std::is_invocable_v<Invocable> || std::is_invocable_v<Invocable, Args...>)&&"invalid arguments");
		}
	}

	template <typename... Args, size_t... Is>
	auto dispatch(std::index_sequence<Is...> idx, Args &&... args) const
	{
		auto result_tuple = std::make_tuple(((invoke(std::get<Is>(actions_), std::forward<Args>(args)...)), ...));

		using tuple_t = decltype(result_tuple);

		if constexpr (is_empty_result_v<tuple_t, Is...>)
		{
			return;
		}
		else if constexpr (num_actions == 1)
		{
			return std::get<0>(result_tuple);
		}
		else
		{
			return result_tuple;
		}
	}
};

template <class... Actions>
sequence(std::tuple<Actions...> actions)->sequence<Actions...>;

template <typename... Actions>
struct sequence_traits<algorithm::sequence<Actions...>>
{
	using is_sequence_type = std::integral_constant<bool, true>;
};

template <template <typename...> typename T, template <typename...> typename U,
		  typename... Ts, typename... Us,
		  typename = std::enable_if_t<is_sequence_v<T<Ts...>> && is_sequence_v<U<Us...>>>>
auto operator|(T<Ts...> &&lhs, T<Us...> &&rhs)
{
	return sequence<Ts..., Us...>{lhs, rhs};
}
} // namespace celerity::algorithm

#endif