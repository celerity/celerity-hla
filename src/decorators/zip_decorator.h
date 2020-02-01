#ifndef ZIP_DECORATOR_H
#define ZIP_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"

namespace celerity::algorithm
{

template <typename TaskType, typename FirstInputValueType, typename SecondInputValueType, typename OutputValueType, int Rank, access_type FirstInputAccessType, access_type SecondInputAccessType>
class zip_task_decorator
{
public:
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto first_input_access_type = FirstInputAccessType;
    static constexpr auto second_input_access_type = SecondInputAccessType;

    using task_type = TaskType;
    using first_input_value_type = FirstInputValueType;
    using second_input_value_type = SecondInputValueType;

    using output_value_type = OutputValueType;
    using first_input_iterator_type = buffer_iterator<first_input_value_type, Rank>;
    using second_input_iterator_type = buffer_iterator<second_input_value_type, Rank>;
    using output_iterator_type = buffer_iterator<output_value_type, Rank>;

    zip_task_decorator(task_type task,
                       first_input_iterator_type in_beg,
                       first_input_iterator_type in_end,
                       second_input_iterator_type second_in_beg,
                       output_iterator_type out_beg)
        : task_(task), in_beg_(in_beg), in_end_(in_end), second_in_beg_(second_in_beg), out_beg_(out_beg)
    {
        assert(in_beg_.get_buffer().get_id() == in_end_.get_buffer().get_id());
        assert(in_beg_.get_buffer().get_id() != out_beg_.get_buffer().get_id());
        assert(second_in_beg_.get_buffer().get_id() != out_beg_.get_buffer().get_id());
    }

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(task_, queue, in_beg_, in_end_);
    }

    task_type get_task() const { return task_; }
    output_iterator_type get_out_iterator() const { return out_beg_; }

private:
    task_type task_;
    first_input_iterator_type in_beg_;
    first_input_iterator_type in_end_;
    second_input_iterator_type second_in_beg_;
    output_iterator_type out_beg_;
};

template <access_type FirstInputAccessType, access_type SecondInputAccessType, typename TaskType, typename FirstInputValueType, typename SecondInputValueType, typename OutputValueType, int Rank>
auto decorate_zip(TaskType task,
                  buffer_iterator<FirstInputValueType, Rank> in_beg,
                  buffer_iterator<FirstInputValueType, Rank> in_end,
                  buffer_iterator<SecondInputValueType, Rank> second_in_beg,
                  buffer_iterator<OutputValueType, Rank> out_beg)
{
    return zip_task_decorator<TaskType, FirstInputValueType, SecondInputValueType, OutputValueType, Rank, FirstInputAccessType, SecondInputAccessType>(
        task, in_beg, in_end, second_in_beg, out_beg);
}

namespace detail
{
template <typename... Args>
struct is_task_decorator<zip_task_decorator<Args...>> : std::bool_constant<true>
{
};
} // namespace detail

} // namespace celerity::algorithm

#endif // ZIP_DECORATOR_H