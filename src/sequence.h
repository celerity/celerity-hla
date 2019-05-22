#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <optional>
#include <variant>
#include <iostream>

#include "celerity.h"
#include "sequence_traits.h"

template<typename... Actions>
class sequence
{
public:
	using actions_t = std::tuple<Actions...>;

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
	void operator()(Args&&...args) const
	{
		dispatch(std::index_sequence_for<Actions...>{}, std::forward<Args>(args)...);
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
	void invoke(const Invocable& invocable, Args&&...args) const
	{
		if constexpr (::is_invocable_v<Invocable, Args...>)
		{
			invocable(std::forward<Args>(args)...);
		}
		else
		{
			invocable();
		}
	}

	template<typename...Args, size_t...Is>
	void dispatch(std::index_sequence<Is...>, Args&&...args) const
	{
		((invoke(std::get<Is>(actions_), std::forward<Args>(args)...)), ...);
	}
};

template<typename...Actions>
struct sequence_traits<sequence<Actions...>>
{
	using is_sequence_type = std::integral_constant<bool, true>;
};

template<template <typename...> typename T, template <typename...> typename U,
	typename...Ts, typename...Us,
	typename = std::enable_if_t<is_sequence_v<T<Ts...>> && is_sequence_v<U<Us...>>>>
	auto operator | (T<Ts...>&& lhs, T<Us...>&& rhs)
{
	return sequence<Ts..., Us...>{ lhs, rhs };
}

#endif 