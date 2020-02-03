#ifndef TRANSFORM_DECORATOR_H
#define TRANSFORM_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../decorator_traits.h"

namespace celerity::algorithm
{

template <int Rank, access_type InputAccessType, typename ComputationFunctor, typename InputValueType, typename OutputValueType>
class packaged_transform
{
public:
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    //using task_type = TaskType;
    using input_value_type = InputValueType;
    using output_value_type = OutputValueType;
    using input_iterator_type = buffer_iterator<input_value_type, Rank>;
    using output_iterator_type = buffer_iterator<output_value_type, Rank>;

    packaged_transform(ComputationFunctor functor, input_iterator_type in_beg, input_iterator_type in_end, output_iterator_type out_beg)
        : functor_(functor), in_beg_(in_beg), in_end_(in_end), out_beg_(out_beg)
    {
        assert(in_beg_.get_buffer().get_id() == in_end_.get_buffer().get_id());
        assert(in_beg_.get_buffer().get_id() != out_beg_.get_buffer().get_id());
    }

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(get_task(), queue, in_beg_, in_end_);
    }

    auto get_task() const
    {
        return functor_(in_beg_, in_end_, out_beg_);
    }

    ComputationFunctor get_computation_functor() const
    {
        return functor_;
    }

    input_iterator_type get_in_beg() const { return in_beg_; }
    input_iterator_type get_in_end() const { return out_beg_; }
    output_iterator_type get_out_iterator() const { return out_beg_; }

private:
    ComputationFunctor functor_;
    input_iterator_type in_beg_;
    input_iterator_type in_end_;
    output_iterator_type out_beg_;
};

template <access_type InputAccessType, typename TaskType, typename InputValueType, typename OutputValueType, int Rank>
auto package_transform(TaskType task, buffer_iterator<InputValueType, Rank> in_beg, buffer_iterator<InputValueType, Rank> in_end, buffer_iterator<OutputValueType, Rank> out_beg)
{
    return packaged_transform<Rank, InputAccessType, TaskType, InputValueType, OutputValueType>(task, in_beg, in_end, out_beg);
}

namespace detail
{

template <int Rank, access_type InputAccessType, typename ComputationFunctor, typename InputValueType, typename OutputValueType>
struct is_packaged_task<packaged_transform<Rank, InputAccessType, ComputationFunctor, InputValueType, OutputValueType>>
    : std::bool_constant<true>
{
};

} // namespace detail

} // namespace celerity::algorithm

#endif // TRANSFORM_DECORATOR_H