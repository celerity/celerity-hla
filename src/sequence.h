#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <optional>
#include <variant>
#include <iostream>

template<typename... Actions>
class Sequence
{
public:
	using actions_t = std::tuple<Actions...>;

	Sequence(Actions... actions)
		: actions_(actions...)
	{

	}

	void operator()() const
	{
		dispatch(std::index_sequence_for<Actions...>{});
	}

private:
	actions_t actions_;

	template<size_t...Is>
	void dispatch(std::index_sequence<Is...>) const
	{
		((std::get<Is>(actions_)()), ...);
	}
};

template<typename...Actions, typename ActionType>
Sequence<Sequence<Actions...>, ActionType> operator | (const Sequence<Actions...>& sequence, ActionType action)
{
	return Sequence<Sequence<Actions...>, ActionType>{ sequence, action };
}

class Eval {};
static constexpr Eval run{};

template<typename T, typename U, typename = std::enable_if_t<std::is_invocable_v<T> && std::is_invocable_v<U>>>
auto operator | (const T& lhs, const U& rhs)
{
	return Sequence<T>{lhs} | rhs;
}

auto hello_world() { return []() { std::cout << "hello world" << std::endl; }; }

auto incr(int& i) { return [&i]() { ++i; }; }

template<typename T>
auto task(const T& invocable)
{ 
	return [invocable]() {
		// queue.submit([=](celerity::handler& cgh) {
		std::invoke(invocable/*, cgh*/);
		// }
	}; 
}

struct Dispatcher { }

Invoker dispatch() { return Dispatcher{}; }

template<typename...Actions>
decltype(auto) operator | (const Sequence<Actions...>& sequence, const Dispatcher& dispatcher)
{
	return std::invoke(sequence);
}

class distr_queue { };

class Runner
{
public:
	Runner(distr_queue q)
		: queue_(q) { }

	void operator()() const {}

private:
	distr_queue queue_;
};

Runner submit_to(distr_queue q)
{
	return Runner(q);
}


