#ifndef TRANSFORM_DECORATOR_H
#define TRANSFORM_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"

namespace celerity::algorithm
{

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

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(task_, queue, in_beg_, in_end_);
    }

    task_type get_task() const { return task_; }
    output_iterator_type get_out_iterator() const { return out_beg_; }

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

} // namespace celerity::algorithm

#endif // TRANSFORM_DECORATOR_H