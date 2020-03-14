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
#include "require.h"

namespace celerity::algorithm
{

namespace detail
{

struct no_result_t
{
};

} // namespace detail

namespace traits
{

template <typename T>
inline constexpr bool is_no_result_v = std::is_same_v<detail::no_result_t, T>;

template <typename T, size_t... Is>
inline constexpr bool is_empty_result_v = (is_no_result_v<std::tuple_element_t<Is, T>> && ...);

} // namespace traits

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

	template <typename... Args, size_t... Is>
	static constexpr bool is_invocable_dispatch(std::index_sequence<Is...>)
	{
		return ((std::is_invocable_v<std::tuple_element_t<Is, actions_t>, Args...>)&&...);
	}

	template <typename... Args>
	decltype(auto) operator()(Args &&... args) const
	{
		static_assert(is_invocable_dispatch<Args...>(std::make_index_sequence<num_actions>{}));

		if constexpr (num_actions == 1 && std::is_void_v<std::invoke_result_t<std::tuple_element_t<0, actions_t>, Args...>>)
		{
			invoke(std::get<0>(actions_), std::forward<Args>(args)...);
			return;
		}
		else
		{
			return dispatch(std::make_index_sequence<num_actions>{}, std::forward<Args>(args)...);
		}
	}

	constexpr actions_t &actions() { return actions_; }
	constexpr const actions_t &actions() const { return actions_; }

private:
	actions_t actions_;

	template <typename... SequenceActions, typename Action, size_t... Ids>
	sequence(sequence<SequenceActions...> &&sequence, Action action, std::index_sequence<Ids...>)
		: actions_(std::move(std::get<Ids>(sequence.actions()))..., action)
	{
	}

	template <typename... Args, size_t... Is>
	auto dispatch(std::index_sequence<Is...> idx, Args &&... args) const
	{
		std::tuple result_tuple = {invoke(std::get<Is>(actions_), std::forward<Args>(args)...)...};

		using tuple_t = decltype(result_tuple);

		if constexpr (traits::is_empty_result_v<tuple_t, Is...>)
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

	template <typename Invocable, typename... Args>
	decltype(auto) invoke(const Invocable &invocable, Args &&... args) const
	{
		if constexpr (std::is_invocable_v<Invocable, Args...>)
		{
			if constexpr (std::is_void_v<std::invoke_result_t<Invocable, Args...>>)
			{
				std::invoke(invocable, std::forward<Args>(args)...);
				return detail::no_result_t{};
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
				return detail::no_result_t{};
			}
			else
			{
				return std::invoke(invocable);
			}
		}
		else
		{
			static_assert(std::is_void_v<Invocable>, "invalid arguments");
		}
	}
};

template <class... Actions>
sequence(std::tuple<Actions...> actions)->sequence<Actions...>;

template <class... Actions, class T>
sequence<Actions..., T> append(const sequence<Actions...> &seq, const T &a)
{
	const auto actions = std::tuple_cat(seq.actions(), std::make_tuple(a));

	return std::apply([](auto &&... args) { return sequence{std::forward<decltype(args)>(args)...}; },
					  actions);
}

namespace traits
{
template <typename... Actions>
struct sequence_traits<algorithm::sequence<Actions...>>
	: std::integral_constant<bool, true>
{
};

template <typename T, typename... Ts>
struct last_element<algorithm::sequence<T, Ts...>>
	: std::tuple_element<algorithm::sequence<T, Ts...>::num_actions - 1,
						 typename algorithm::sequence<T, Ts...>::actions_t>
{
};

template <>
struct last_element<algorithm::sequence<>>
{
	using type = void;
};

template <typename T, typename... Ts>
struct first_element<algorithm::sequence<T, Ts...>>
	: std::tuple_element<0, typename algorithm::sequence<T, Ts...>::actions_t>
{
};

template <>
struct first_element<algorithm::sequence<>>
{
	using type = void;
};

} // namespace traits

template <typename... Ts, typename... Us>
auto operator|(sequence<Ts...> lhs, sequence<Us...> rhs)
{
	return sequence<Ts..., Us...>{std::tuple_cat(lhs.actions(), rhs.actions())};
}

template <typename T,
		  require<traits::is_sequence_v<T>> = yes>
constexpr auto get_last_element(const T &s)
{
	return std::get<traits::size_v<T> - 1>(s.actions());
}

template <typename T, size_t... Is>
constexpr auto dispatch_remove_last_element(const T &s, std::index_sequence<Is...>)
{
	return sequence(std::get<Is>(s.actions())...);
}

template <typename T,
		  require<traits::is_sequence_v<T>> = yes>
constexpr auto remove_last_element(const T &s)
{
	return dispatch_remove_last_element(s, std::make_index_sequence<traits::size_v<T> - 1>{});
}

template <typename T,
		  require<traits::is_sequence_v<T>> = yes>
constexpr auto get_first_element(const T &s)
{
	return std::get<0>(s.actions());
}

template <typename T, size_t... Is>
constexpr auto dispatch_remove_first_element(const T &s, std::index_sequence<Is...>)
{
	return sequence(std::get<Is + 1>(s.actions())...);
}

template <typename T,
		  require<traits::is_sequence_v<T>> = yes>
constexpr auto remove_first_element(const T &s)
{
	return dispatch_remove_first_element(s, std::make_index_sequence<traits::size_v<T> - 1>{});
}

template <class... Actions, class T>
auto apply_append(const sequence<Actions...> &seq, const T &a)
{
	return remove_last_element(seq) | (get_last_element(seq) | a);
}

} // namespace celerity::algorithm

#endif