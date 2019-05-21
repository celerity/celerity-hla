#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <optional>
#include <variant>
#include <iostream>

template <typename F, typename... Args>
struct is_invocable :
    std::is_constructible<
        std::function<void(Args ...)>,
        std::reference_wrapper<typename std::remove_reference<F>::type>
    >
{
};

template <typename F, typename... Args>
constexpr inline bool is_invocable_v = is_invocable<F, Args...>::value;

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

  actions_t& actions() { return actions_; }      

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
    if constexpr (is_invocable_v<Invocable, Args...>)
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

template<template <typename...> typename Sequence, typename...Actions, typename Action>
auto operator | (Sequence<Actions...>&& seq, Action action)
{
	return sequence<Actions..., Action>{ std::move(seq), action };
}

template<template <typename...> typename T, typename...Ts,          
         template <typename...> typename U, typename...Us>
auto operator | (T<Ts...>&& lhs, T<Us...>&& rhs)
{
	return sequence<T<Ts...>, U<Us...>>{ lhs, rhs };
}

template<typename T, typename U/*, typename = std::enable_if_t<std::is_invocable_v<T> && std::is_invocable_v<U>>*/>
auto operator | (const T& lhs, const U& rhs)
{
	return sequence<T, U>{lhs, rhs};
}

#endif 