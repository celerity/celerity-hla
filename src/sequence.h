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
	template<typename... Actions>
	class sequence
	{
	public:
		using actions_t = std::tuple<Actions...>;
		static constexpr auto num_actions = sizeof...(Actions);

		sequence(Actions... actions)
			: actions_(actions...)
		{

		}

		template<typename...SequenceActions, typename Action>
		sequence(sequence<SequenceActions...>&& seq, Action action)
			: sequence(std::move(seq), action, std::index_sequence_for<SequenceActions...>{})
		{

		}

		template<typename...Args>
		decltype(auto) operator()(Args&& ...args) const
		{
			if constexpr (num_actions > 1)
			{
				dispatch(std::make_index_sequence<num_actions - 1>{}, std::forward<Args>(args)...);
			}

			return invoke(std::get<num_actions - 1>(actions_), std::forward<Args>(args)...);
		}

		constexpr actions_t& actions() { return actions_; }

	private:
		actions_t actions_;

		template<typename...SequenceActions, typename Action, size_t...Ids>
		sequence(sequence<SequenceActions...>&& sequence, Action action, std::index_sequence<Ids...>)
			: actions_(std::move(std::get<Ids>(sequence.actions()))..., action)
		{
		}

		template<typename Invocable, typename...Args>
		decltype(auto) invoke(const Invocable& invocable, Args&& ...args) const
		{
			if constexpr (std::is_invocable_v<Invocable, Args...>)
			{
				if constexpr (std::is_void_v<std::invoke_result_t<Invocable, Args...>>)
				{
					std::invoke(invocable, std::forward<Args>(args)...);
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
				}
				else
				{
					return std::invoke(invocable);
				}
			}
			else
			{
				assert((std::is_invocable_v<Invocable> || std::is_invocable_v<Invocable, Args...>) && "invalid arguments");
			}
		}

		template<typename...Args, size_t...Is>
		void dispatch(std::index_sequence<Is...>, Args&& ...args) const
		{
			((invoke(std::get<Is>(actions_), std::forward<Args>(args)...)), ...);
		}
	};

	template<typename...Actions>
	struct sequence_traits<algorithm::sequence<Actions...>>
	{
		using is_sequence_type = std::integral_constant<bool, true>;
	};

	template<template <typename...> typename T, template <typename...> typename U,
		typename...Ts, typename...Us,
		typename = std::enable_if_t<is_sequence_v<T<Ts...>> && is_sequence_v<U<Us...>>>>
		auto operator | (T<Ts...> && lhs, T<Us...> && rhs)
	{
		return sequence<Ts..., Us...>{ lhs, rhs };
	}
}

#endif 