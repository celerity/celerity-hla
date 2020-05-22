#ifndef T_JOINT_H
#define T_JOINT_H

#include "packaged_task_traits.h"

namespace celerity::algorithm
{
namespace detail
{
template <typename T>
constexpr inline bool is_sequence_list_v = traits::is_sequence_v<T>&& all_of<T>(
    [](auto e) { return std::bool_constant<traits::is_sequence_v<decltype(e)>>{}; });

template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct t_joint;

template <typename Task, typename SecondaryInputSequence>
struct t_joint<Task, SecondaryInputSequence, false>
{
  public:
	static_assert(traits::is_sequence_v<SecondaryInputSequence> && !is_sequence_list_v<SecondaryInputSequence>);

	t_joint(Task task, SecondaryInputSequence sequence) : task_(task), secondary_in_(sequence) {}

	auto operator()(celerity::distr_queue& queue) const
	{
		std::invoke(secondary_in_, queue);
		return std::invoke(task_, queue);
	}

	auto get_in_beg() const { return task_.get_in_beg(); }
	auto get_in_end() const { return task_.get_in_end(); }
	auto get_out_beg() const { return task_.get_out_beg(); }
	auto get_range() const { return task_.get_range(); }

	auto get_task() { return task_; }
	auto get_secondary() { return secondary_in_; }

  private:
	Task task_;
	SecondaryInputSequence secondary_in_;
};

template <typename Task, typename SequenceList>
struct t_joint<Task, SequenceList, true>
{
  public:
	static_assert(is_sequence_list_v<SequenceList>);

	t_joint(Task task, SequenceList sequences) : task_(task), sequences_(sequences) {}

	auto operator()(celerity::distr_queue& queue) const
	{
		std::invoke(sequences_, queue);
		return std::invoke(task_, queue);
	}

	auto get_in_beg() const { return task_.get_in_beg(); }
	auto get_in_end() const { return task_.get_in_end(); }
	auto get_out_beg() const { return task_.get_out_beg(); }
	auto get_range() const { return task_.get_range(); }

	auto get_task() { return task_; }

  private:
	Task task_;
	SequenceList sequences_;
};

template <typename Task, typename SecondaryInputSequence>
struct partial_t_joint
{
  public:
	static_assert(!traits::is_sequence_v<Task>);

	partial_t_joint(Task task, SecondaryInputSequence sequence) : task_(task), secondary_in_(sequence) {}

	template <typename IteratorType>
	auto complete(IteratorType beg, IteratorType end)
	{
		auto completed_task = task_.complete(beg, end);

		using completed_task_type = decltype(completed_task);

		if constexpr(traits::is_partially_packaged_task_v<completed_task_type>)
		{ return partial_t_joint<completed_task_type, SecondaryInputSequence>{completed_task, secondary_in_}; }
		else
		{
			return t_joint<completed_task_type, SecondaryInputSequence, false>{completed_task, secondary_in_};
		}
	}

	auto get_in_beg() const { return task_.get_in_beg(); }
	auto get_in_end() const { return task_.get_in_end(); }
	auto get_range() const { return task_.get_range(); }
	auto get_task() const { return task_; }
	auto get_secondary() const { return secondary_in_; }

  private:
	Task task_;
	SecondaryInputSequence secondary_in_;
};

template <typename Task, typename Sequence>
auto make_t_joint(Task t, Sequence s)
{
	return t_joint<Task, Sequence, is_sequence_list_v<Sequence>>(t, s);
}

template <typename Task, typename Sequence>
auto make_partial_t_joint(Task t, Sequence s)
{
	return partial_t_joint<Task, Sequence>(t, s);
}

} // namespace detail

namespace traits
{
template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct is_packaged_task<detail::t_joint<Task, SecondaryInputSequence, SequenceList>> : std::bool_constant<true>
{
};

template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct packaged_task_traits<detail::t_joint<Task, SecondaryInputSequence, SequenceList>>
{
	using traits = packaged_task_traits<Task>;

	static constexpr auto rank = traits::rank;
	static constexpr auto computation_type = traits::computation_type;
	static constexpr auto access_type = traits::access_type;

	using input_iterator_type = typename traits::input_iterator_type;
	using input_value_type = typename traits::input_value_type;
	using output_value_type = typename traits::output_value_type;
	using output_iterator_type = typename traits::output_iterator_type;
};

template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct extended_packaged_task_traits<detail::t_joint<Task, SecondaryInputSequence, SequenceList>, detail::computation_type::zip>
{
	using traits = extended_packaged_task_traits<Task, detail::computation_type::zip>;

	static constexpr auto second_input_access_type = traits::second_input_access_type;

	using second_input_value_type = typename traits::second_input_value_type;
	using second_input_iterator_type = typename traits::second_input_iterator_type;
};

template <typename Task, typename SecondaryInputSequence>
struct is_partially_packaged_task<detail::partial_t_joint<Task, SecondaryInputSequence>> : std::bool_constant<true>
{
};

template <typename Task, typename SecondaryInputSequence>
struct packaged_task_traits<detail::partial_t_joint<Task, SecondaryInputSequence>>
{
	using traits = packaged_task_traits<Task>;

	static constexpr auto rank = traits::rank;
	static constexpr auto computation_type = traits::computation_type;
	static constexpr auto access_type = traits::access_type;

	using input_iterator_type = typename traits::input_iterator_type;
	using input_value_type = typename traits::input_value_type;
	using output_value_type = typename traits::output_value_type;
	using output_iterator_type = typename traits::output_iterator_type;
};

template <typename Task, typename SecondaryInputSequence>
struct extended_packaged_task_traits<detail::partial_t_joint<Task, SecondaryInputSequence>, detail::computation_type::zip>
    : extended_packaged_task_traits<Task, detail::computation_type::zip>
{
};

template <typename Task, typename SecondaryInputSequence>
struct partially_packaged_task_traits<detail::partial_t_joint<Task, SecondaryInputSequence>> : partially_packaged_task_traits<Task>
{
};

template <typename T>
struct is_t_joint : std::bool_constant<false>
{
};

template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct is_t_joint<detail::t_joint<Task, SecondaryInputSequence, SequenceList>> : std::bool_constant<true>
{
};

template <typename Task, typename SecondaryInputSequence>
struct is_t_joint<detail::partial_t_joint<Task, SecondaryInputSequence>> : std::bool_constant<true>
{
};

template <typename T>
constexpr inline bool is_t_joint_v = is_t_joint<T>::value;

template <typename T>
struct t_joint_traits
{
	using task_type = void;
	using secondary_input_sequence_type = detail::sequence<>;
};

template <typename Task, typename SecondaryInputSequence, bool SequenceList>
struct t_joint_traits<detail::t_joint<Task, SecondaryInputSequence, SequenceList>>
{
	using task_type = Task;
	using secondary_input_sequence_type = SecondaryInputSequence;
};

template <typename Task, typename SecondaryInputSequence>
struct t_joint_traits<detail::partial_t_joint<Task, SecondaryInputSequence>>
{
	using task_type = Task;
	using secondary_input_sequence_type = SecondaryInputSequence;
};

template <typename T>
using secondary_sequence_t = typename t_joint_traits<T>::secondary_input_sequence_type;

} // namespace traits

} // namespace celerity::algorithm

#endif // T_JOINT_H