#ifndef GENERATE_DECORATOR_H
#define GENERATE_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"

namespace celerity::algorithm
{

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

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(task_, queue, out_beg_, out_end_);
    }

    output_iterator_type get_out_iterator() const { return out_beg_; }
    task_type get_task() const { return task_; }

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

namespace detail
{

template <typename... Args>
struct is_task_decorator<generate_task_decorator<Args...>> : std::bool_constant<true>
{
};

} // namespace detail

} // namespace celerity::algorithm

#endif // GENERATE_DECORATOR_H