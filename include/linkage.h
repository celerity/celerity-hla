#ifndef LINKAGE_H
#define LINKAGE_H

#include "depth_first.h"
#include "linkage_traits.h"
#include "require.h"
#include "transient.h"

namespace celerity::algorithm::detail
{
template <typename... Actions>
auto link(const sequence<Actions...>& s);

namespace link_impl
{
template <typename T, typename U>
auto link_transiently(T lhs, U rhs)
{
	using value_type = typename traits::packaged_task_traits<T>::output_value_type;
	constexpr auto rank = traits::packaged_task_traits<T>::rank;

	transient_buffer<value_type, rank> out_buf{lhs.get_range()};

	auto t_left = lhs.complete(begin(out_buf), end(out_buf));
	auto t_right = rhs.complete(begin(out_buf), end(out_buf));

	return sequence(t_left, t_right);
}

template <typename T, typename U>
auto link(T lhs, U rhs)
{
	using value_type = typename traits::packaged_task_traits<T>::output_value_type;
	constexpr auto rank = traits::packaged_task_traits<T>::rank;

	buffer<value_type, rank> out_buf{lhs.get_range()};

	auto t_left = lhs.complete(begin(out_buf), end(out_buf));
	auto t_right = rhs.complete(begin(out_buf), end(out_buf));

	return sequence(t_left, t_right);
}

template <typename T, require<!traits::is_internally_linked_v<T>> = yes>
auto link_internally(T t_joint)
{
	using namespace traits;

	auto secondary = t_joint.get_secondary();
	auto last = get_last_element(secondary);

	if constexpr(is_celerity_buffer_v<decltype(last)>)
	{
		if constexpr(size_v<decltype(secondary)> == 1) { return t_joint.get_task().complete(begin(last), end(last)); }
		else
		{
			return t_joint.complete(begin(last), end(last));
		}
	}
	else if constexpr(is_transiently_linkable_source_v<decltype(last)> && has_transiently_linkable_second_input_v<T>)
	{
		const auto linked = link_transiently(get_last_element(secondary), t_joint.get_task());

		return partial_t_joint{get_last_element(linked), append(remove_last_element(secondary), get_first_element(linked))};
	}
	else if constexpr(is_linkable_source_v<decltype(last)> && is_linkable_sink_v<T>)
	{
		const auto linked = link(get_last_element(secondary), t_joint.get_task());

		return partial_t_joint{get_last_element(linked), append(remove_last_element(secondary), get_first_element(linked))};
	}
	else
	{
		static_assert(std::is_void_v<T>, "logic error");
	}
}

template <typename T, require<traits::is_internally_linked_v<T>> = yes>
auto link_internally(T task)
{
	return task;
}

template <typename T, typename U, require<traits::is_linkable_source_v<T>, traits::is_linkable_sink_v<U>> = yes>
auto operator+(T lhs, U rhs)
{
	using namespace traits;
	using namespace detail;

	auto l = link_internally(lhs);
	auto r = link_internally(rhs);

	if constexpr(is_transiently_linkable_source_v<decltype(l)> && has_transiently_linkable_first_input_v<decltype(r)>) { return link_transiently(l, r); }
	else if constexpr(is_linkable_source_v<decltype(l)> && is_linkable_sink_v<decltype(r)>)
	{
		return link(l, r);
	}
	else
	{
		return sequence(l, r);
	}
}

template <typename T, typename U, require<traits::is_packaged_task_v<T>, traits::is_linkable_sink_v<U>> = yes>
auto operator+(T lhs, U rhs)
{
	auto t_right = link_internally(rhs).complete(lhs.get_out_iterator(), lhs.get_out_iterator());
	return detail::sequence(lhs, t_right);
}

template <typename T, typename U, require<!traits::is_linkable_source_v<T>, !traits::is_linkable_sink_v<U>> = yes>
auto operator+(T lhs, U rhs)
{
	return detail::sequence(lhs, rhs);
}

template <typename T, int Rank, typename U, require<algorithm::traits::is_linkable_sink_v<U>> = yes>
auto operator+(const celerity::buffer<T, Rank>& lhs, U rhs)
{
	return detail::sequence(link_internally(rhs).complete(begin(lhs), end(lhs)));
}

template <typename T, int Rank, typename U, require<algorithm::traits::is_linkable_source_v<U>> = yes>
auto operator+(U lhs, const celerity::buffer<T, Rank>& rhs)
{
	return detail::sequence(link_internally(lhs).complete(begin(rhs), end(rhs)));
}

template <typename T, typename U,
    require<algorithm::traits::is_sequence_v<T>, algorithm::traits::is_linkable_source_v<traits::last_element_t<T>>, algorithm::traits::is_linkable_sink_v<U>> =
        yes>
auto operator+(T lhs, U rhs)
{
	constexpr auto op = [](auto&& a, auto&& b) { return link_impl::operator+(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); };

	return apply_append(lhs, rhs, op);
}

template <typename... Actions, size_t... Is>
auto link(const sequence<Actions...>& s, std::index_sequence<Is...>)
{
	constexpr auto op = [](auto&& a, auto&& b) { return link_impl::operator+(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)); };

	return left_fold(s, op);
}

namespace post_conditions
{
template <typename T>
constexpr inline bool element_post_condition()
{
	return !traits::is_celerity_buffer_v<T>;
}

template <typename T>
constexpr T ensure(T s)
{
	using namespace traits;
	static_assert(!traits::is_packaged_task_sequence_v<T>);
	return s;
}

} // namespace post_conditions

} // namespace link_impl

template <typename... Actions>
auto link(const sequence<Actions...>& s)
{
	using link_impl::link;
	using link_impl::link_internally;
	using link_impl::post_conditions::ensure;

	// TODO
	const auto seq = traverse(s, [](const auto& seq) { return link(seq); });
	// const auto seq = s;

	static_assert(traits::size_v<decltype(s)> == traits::size_v<decltype(seq)>);

	return ensure([&]() {
		if constexpr(traits::size_v<decltype(seq)> == 1) { return detail::sequence(link_internally(get_first_element(seq))); }
		else
		{
			return link(seq, std::make_index_sequence<traits::size_v<decltype(seq)>>{});
		}
	}());
}

template <typename T, require<traits::is_partially_packaged_task_v<T>, traits::stage_requirement_v<T> == detail::stage_requirement::output> = yes>
auto terminate(T task)
{
	using traits = traits::packaged_task_traits<T>;
	using value_type = typename traits::output_value_type;

	constexpr auto rank = traits::rank;

	buffer<value_type, rank> out_buf{task.get_range()};

	return sequence(task.complete(begin(out_buf), end(out_buf)));
}

template <typename T, require<!traits::is_partially_packaged_task_v<T>, !traits::is_sequence_v<T>> = yes>
auto terminate(T task)
{
	return sequence(task);
}

template <typename T, require<traits::is_sequence_v<T>, traits::is_partially_packaged_task_v<traits::last_element_t<T>>,
                          traits::stage_requirement_v<traits::last_element_t<T>> == detail::stage_requirement::output> = yes>
auto terminate(T seq)
{
	return append(remove_last_element(seq), get_first_element(terminate(get_last_element(seq))));
}

} // namespace celerity::algorithm::detail

#endif // LINKAGE_H