#ifndef TRANSFORM_DECORATOR_H
#define TRANSFORM_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../packaged_task_traits.h"
#include "../partially_packaged_task.h"
#include "../accessor_type.h"

namespace celerity::algorithm
{

namespace detail
{
template <int Rank, detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
class packaged_transform
{
public:
    static_assert(!std::is_void_v<typename std::iterator_traits<InputIteratorType>::value_type>);
    static_assert(!std::is_void_v<typename std::iterator_traits<OutputIteratorType>::value_type>);

    packaged_transform(Functor functor, InputIteratorType in_beg, InputIteratorType in_end, OutputIteratorType out_beg)
        : functor_(functor), in_beg_(in_beg), in_end_(in_end), out_beg_(out_beg)
    {
        assert(are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
        assert(!are_equal(in_beg_.get_buffer(), out_beg_.get_buffer()));
        //assert(distance(in_beg_, in_end_) <= distance(out_beg_, end(out_beg_.get_buffer())));
    }

    auto operator()(celerity::distr_queue &queue) const
    {
        std::invoke(get_task(), queue, in_beg_, in_end_);
        return out_beg_.get_buffer();
    }

    auto get_task() const { return functor_; }

    InputIteratorType get_in_beg() const { return in_beg_; }
    InputIteratorType get_in_end() const { return in_end_; }
    OutputIteratorType get_out_iterator() const { return out_beg_; }

private:
    Functor functor_;
    InputIteratorType in_beg_;
    InputIteratorType in_end_;
    OutputIteratorType out_beg_;
};

template <detail::access_type InputAccessType, typename FunctorType, template <typename, int> typename InIteratorType, template <typename, int> typename OutIteratorType, typename InputValueType, typename OutputValueType, int Rank>
auto package_transform(FunctorType task,
                       InIteratorType<InputValueType, Rank> in_beg,
                       InIteratorType<InputValueType, Rank> in_end,
                       OutIteratorType<OutputValueType, Rank> out_beg)
{
    return packaged_transform<Rank, InputAccessType, FunctorType, InIteratorType<InputValueType, Rank>, OutIteratorType<OutputValueType, Rank>>(
        task, in_beg, in_end, out_beg);
}

template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputIteratorType>
class partially_packaged_transform_1
{
public:
    static_assert(!std::is_void_v<typename std::iterator_traits<InputIteratorType>::value_type>);

    explicit partially_packaged_transform_1(Functor f, InputIteratorType beg, InputIteratorType end)
        : f_(f), in_beg_(beg), in_end_(end) {}

    template <typename Iterator>
    auto complete(Iterator beg, Iterator) const
    {
        const auto f = std::invoke(f_, in_beg_, in_end_, beg);
        return package_transform<InputAccessType>(f, in_beg_, in_end_, beg);
    }

    InputIteratorType get_in_beg() const { return in_beg_; }
    InputIteratorType get_in_end() const { return in_end_; }

    cl::sycl::range<Rank> get_range() const { return distance(in_beg_, in_end_); }

private:
    Functor f_;
    InputIteratorType in_beg_;
    InputIteratorType in_end_;
};

template <detail::access_type InputAccessType, typename KernelFunctor, typename FunctorType, int Rank, template <typename, int> typename InIteratorType, typename InputValueType>
auto package_transform(FunctorType functor, InIteratorType<InputValueType, Rank> beg, InIteratorType<InputValueType, Rank> end)
{
    return partially_packaged_transform_1<Rank, InputAccessType, FunctorType, KernelFunctor, InIteratorType<InputValueType, Rank>>(functor, beg, end);
}

template <detail::access_type InputAccessType, typename Functor, typename KernelFunctor>
class partially_packaged_transform_0
{
public:
    explicit partially_packaged_transform_0(Functor f)
        : f_(f) {}

    template <typename Iterator>
    auto complete(Iterator beg, Iterator end) const
    {
        return package_transform<InputAccessType, KernelFunctor>(f_, beg, end);
    }

private:
    Functor f_;
};

template <access_type InputAccessType, typename KernelFunctor, typename FunctorType>
auto package_transform(FunctorType functor)
{
    return partially_packaged_transform_0<InputAccessType, FunctorType, KernelFunctor>(functor);
}

} // namespace detail

namespace traits
{

template <int Rank, detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
struct is_packaged_task<detail::packaged_transform<Rank, InputAccessType, Functor, InputIteratorType, OutputIteratorType>>
    : std::true_type
{
};

template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputValueType>
struct is_partially_packaged_task<detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputValueType>>
    : std::true_type
{
};

template <detail::access_type InputAccessType, typename Functor, typename KernelFunctor>
struct is_partially_packaged_task<detail::partially_packaged_transform_0<InputAccessType, Functor, KernelFunctor>>
    : std::true_type
{
};

template <int Rank, detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
struct packaged_task_traits<detail::packaged_transform<Rank, InputAccessType, Functor, InputIteratorType, OutputIteratorType>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = detail::computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    using input_iterator_type = InputIteratorType;
    using input_value_type = typename std::iterator_traits<InputIteratorType>::value_type;
    using output_value_type = typename std::iterator_traits<OutputIteratorType>::value_type;
    using output_iterator_type = OutputIteratorType;
};

template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputIteratorType>
struct packaged_task_traits<detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputIteratorType>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = detail::computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    using input_iterator_type = InputIteratorType;
    using input_value_type = typename std::iterator_traits<InputIteratorType>::value_type;
    using output_value_type = kernel_result_t<KernelFunctor>;
    using output_iterator_type = void;
};

template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputIteratorType>
struct partially_packaged_task_traits<detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputIteratorType>>
{
    static constexpr auto requirement = detail::stage_requirement::output;
};

template <detail::access_type InputAccessType, typename Functor, typename KernelFunctor>
struct packaged_task_traits<detail::partially_packaged_transform_0<InputAccessType, Functor, KernelFunctor>>
{
    static constexpr auto rank = -1;
    static constexpr auto computation_type = detail::computation_type::transform;
    static constexpr auto access_type = InputAccessType;

    using input_iterator_type = void;
    using input_value_type = void;
    using output_value_type = kernel_result_t<KernelFunctor>;
    using output_iterator_type = void;
};

template <detail::access_type InputAccessType, typename Functor, typename KernelFunctor>
struct partially_packaged_task_traits<detail::partially_packaged_transform_0<InputAccessType, Functor, KernelFunctor>>
{
    static constexpr auto requirement = detail::stage_requirement::input;
};

} // namespace traits

} // namespace celerity::algorithm

#endif // TRANSFORM_DECORATOR_H