#ifndef DECORATED_TASK_H
#define DECORATED_TASK_H

#include "iterator.h"
#include "celerity_helper.h"

namespace celerity::algorithm
{

enum class computation_type
{
    generate,
    transform,
    reduce,
    zip
};

template <typename TaskType, typename InputValueType, typename OutputValueType, int Rank, access_type InputAccessType>
class transform_task_decorator
{
public:
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    using task_type = TaskType;
    using input_value_type = InputValueType;
    using output_value_type = OutputValueType;
    using input_iterator_type = buffer_iterator<input_value_type, Rank>;
    using output_iterator_type = buffer_iterator<output_value_type, Rank>;

    transform_task_decorator(task_type task, input_iterator_type in_beg, input_iterator_type in_end, output_iterator_type out_beg)
        : task_(task), in_beg_(in_beg), in_end_(in_end), out_beg_(out_beg)
    {
        assert(in_beg_.get_buffer().get_id() == in_end_.get_buffer().get_id());
        assert(in_beg_.get_buffer().get_id() != out_beg_.get_buffer().get_id());
    }

    void operator()(celerity::distr_queue &queue)
    {
        std::invoke(task_, queue, in_beg_, in_end_);
    }

private:
    task_type task_;
    input_iterator_type in_beg_;
    input_iterator_type in_end_;
    output_iterator_type out_beg_;
};

template <access_type InputAccessType, typename TaskType, typename InputValueType, typename OutputValueType, int Rank>
auto decorate_transform(TaskType task, buffer_iterator<InputValueType, Rank> in_beg, buffer_iterator<InputValueType, Rank> in_end, buffer_iterator<OutputValueType, Rank> out_beg)
{
    return transform_task_decorator<TaskType, InputValueType, OutputValueType, Rank, InputAccessType>(task, in_beg, in_end, out_beg);
}

template <typename TaskType, typename InputValueType, typename OutputValueType, int Rank>
class generate_task_decorator
{
public:
    static constexpr auto computation_type = computation_type::generate;

    using task_type = TaskType;
    using input_value_type = InputValueType;
    using output_value_type = OutputValueType;
    using output_iterator_type = buffer_iterator<output_value_type, Rank>;

    static_assert(std::is_convertible_v<input_value_type, output_value_type>);

    generate_task_decorator(task_type task, output_iterator_type out_beg, output_iterator_type out_end)
        : task_(task), out_beg_(out_beg), out_end_(out_end)
    {
        assert(out_beg_.get_buffer().get_id() == out_end_.get_buffer().get_id());
    }

    void operator()(celerity::distr_queue &queue)
    {
        std::invoke(task_, queue, out_beg_, out_end_);
    }

private:
    task_type task_;
    output_iterator_type out_beg_;
    output_iterator_type out_end_;
};

template <typename InputValueType, typename TaskType, typename OutputValueType, int Rank>
auto decorate_generate(TaskType task, buffer_iterator<OutputValueType, Rank> out_beg, buffer_iterator<OutputValueType, Rank> out_end)
{
    return generate_task_decorator<TaskType, InputValueType, OutputValueType, Rank>(task, out_beg, out_end);
}

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

    void operator()(celerity::distr_queue &queue)
    {
        std::invoke(task_, queue, in_beg_, in_end_);
    }

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

} // namespace celerity::algorithm

#endif // DECORATED_TASK_H