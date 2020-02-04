#ifndef TRANSFORM_DECORATOR_H
#define TRANSFORM_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../packaged_task_traits.h"
#include "partially_packaged_task.h"
#include "../accessor_type.h"

namespace celerity::algorithm
{

template <int Rank, access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputValueType>
class partially_packaged_transform_1;

template <access_type InputAccessType, typename Functor, typename KernelFunctor>
class partially_packaged_transform_0;

template <int Rank, access_type InputAccessType, typename Functor, typename InputValueType, typename OutputValueType>
class packaged_transform
{
public:
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    using functor_type = Functor;
    using input_value_type = InputValueType;
    using output_value_type = OutputValueType;
    using input_iterator_type = buffer_iterator<input_value_type, Rank>;
    using output_iterator_type = buffer_iterator<output_value_type, Rank>;

    packaged_transform(Functor functor, input_iterator_type in_beg, input_iterator_type in_end, output_iterator_type out_beg)
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

    Functor get_computation_functor() const
    {
        return functor_;
    }

    input_iterator_type get_in_beg() const { return in_beg_; }
    input_iterator_type get_in_end() const { return in_end_; }
    output_iterator_type get_out_iterator() const { return out_beg_; }

    /*auto repackage(input_iterator_type in_beg, input_iterator_type in_end, output_iterator_type out_beg)
    {
        return packaged_transform<Rank, access_type>(functor_, in_beg, in_end, out_beg);
    }

    auto repackage_partially(input_iterator_type in_beg, input_iterator_type in_end)
    {
        return partially_packaged_transform_1<Rank, access_type>(functor_, in_beg, in_end);
    }

    auto unpackage()
    {
        return partially_packaged_transform_0<Rank, access_type>(functor_);
    }*/

private:
    functor_type functor_;
    input_iterator_type in_beg_;
    input_iterator_type in_end_;
    output_iterator_type out_beg_;
};

template <access_type InputAccessType, typename FunctorType, typename InputValueType, typename OutputValueType, int Rank>
auto package_transform(FunctorType task, buffer_iterator<InputValueType, Rank> in_beg, buffer_iterator<InputValueType, Rank> in_end, buffer_iterator<OutputValueType, Rank> out_beg)
{
    return packaged_transform<Rank, InputAccessType, FunctorType, InputValueType, OutputValueType>(task, in_beg, in_end, out_beg);
}

template <int Rank, access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputValueType>
class partially_packaged_transform_1
{
public:
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto access_type = InputAccessType;
    static constexpr auto requirement = stage_requirement::output;

    using functor_type = Functor;
    using kernel_functor_type = KernelFunctor;
    using input_value_type = InputValueType;
    using input_iterator_type = buffer_iterator<input_value_type, Rank>;
    using output_value_type = detail::kernel_result_t<kernel_functor_type, input_value_type>;

    explicit partially_packaged_transform_1(Functor f, input_iterator_type beg, input_iterator_type end)
        : f_(f), in_beg_(beg), in_end_(end) {}

    template <typename Iterator>
    auto complete(Iterator beg, Iterator) const
    {
        return package_transform<access_type>(f_, in_beg_, in_end_, beg);
    }

    input_iterator_type get_in_beg() const { return in_beg_; }
    input_iterator_type get_in_end() const { return in_end_; }

private:
    functor_type f_;
    input_iterator_type in_beg_;
    input_iterator_type in_end_;
};

template <access_type InputAccessType, typename KernelFunctor, typename FunctorType, int Rank, typename InputValueType>
auto package_transform(FunctorType functor, buffer_iterator<InputValueType, Rank> beg, buffer_iterator<InputValueType, Rank> end)
{
    return partially_packaged_transform_1<Rank, InputAccessType, FunctorType, KernelFunctor, InputValueType>(functor, beg, end);
}

template <access_type InputAccessType, typename Functor, typename KernelFunctor>
class partially_packaged_transform_0
{
public:
    static constexpr auto computation_type = computation_type::transform;
    static constexpr auto access_type = InputAccessType;
    static constexpr auto requirement = stage_requirement::input;

    using functor_type = Functor;
    using kernel_functor_type = KernelFunctor;

    explicit partially_packaged_transform_0(Functor f)
        : f_(f) {}

    template <typename Iterator>
    auto complete(Iterator beg, Iterator end) const
    {
        return package_transform<InputAccessType, KernelFunctor>(f_, beg, end);
    }

private:
    functor_type f_;
};

template <access_type InputAccessType, typename KernelFunctor, typename FunctorType>
auto package_transform(FunctorType functor)
{
    return partially_packaged_transform_0<InputAccessType, FunctorType, KernelFunctor>(functor);
}

namespace detail
{

template <int Rank, access_type InputAccessType, typename Functor, typename InputValueType, typename OutputValueType>
struct is_packaged_task<packaged_transform<Rank, InputAccessType, Functor, InputValueType, OutputValueType>>
    : std::bool_constant<true>
{
};

template <int Rank, access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputValueType>
struct is_partially_packaged_task<partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputValueType>>
    : std::bool_constant<true>
{
};

template <access_type InputAccessType, typename Functor, typename KernelFunctor>
struct is_partially_packaged_task<partially_packaged_transform_0<InputAccessType, Functor, KernelFunctor>>
    : std::bool_constant<true>
{
};

} // namespace detail

} // namespace celerity::algorithm

#endif // TRANSFORM_DECORATOR_H