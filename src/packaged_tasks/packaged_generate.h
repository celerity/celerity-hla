#ifndef PACKAGED_GENERATE_H
#define PACKAGED_GENERATE_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../packaged_task_traits.h"

namespace celerity::algorithm
{

template <typename FunctorType, typename InputValueType, typename OutputIteratorType, int Rank, bool Fused>
class packaged_generate
{
public:
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::generate;

    using functor_type = FunctorType;
    using input_value_type = InputValueType;
    using output_value_type = typename std::iterator_traits<OutputIteratorType>::value_type;
    using output_iterator_type = OutputIteratorType;

    static_assert(std::is_convertible_v<input_value_type, output_value_type>);

    packaged_generate(functor_type functor, output_iterator_type out_beg, output_iterator_type out_end)
        : functor_(functor), out_beg_(out_beg), out_end_(out_end)
    {
        assert(out_beg_.get_buffer().get_id() == out_end_.get_buffer().get_id());
    }

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(get_task(), queue, out_beg_, out_end_);
    }

    output_iterator_type get_out_iterator() const { return out_beg_; }

    auto get_task() const 
    { 
        if constexpr(Fused)
        {
            return functor_; 
        }
        else
        {
            return functor_(out_beg_, out_end_);
        }
    }

private:
    functor_type functor_;
    output_iterator_type out_beg_;
    output_iterator_type out_end_;
};

template <typename InputValueType, typename FunctorType, template <typename, int> typename OutIteratorType, typename OutputValueType, int Rank>
auto package_generate(FunctorType functor, OutIteratorType<OutputValueType, Rank> out_beg, OutIteratorType<OutputValueType, Rank> out_end)
{
    return packaged_generate<FunctorType, InputValueType, OutIteratorType<OutputValueType, Rank>, Rank, false>(functor, out_beg, out_end);
}

template <typename InputValueType, bool Fused, typename FunctorType, template <typename, int> typename OutIteratorType, typename OutputValueType, int Rank>
auto package_generate(FunctorType functor, OutIteratorType<OutputValueType, Rank> out_beg, OutIteratorType<OutputValueType, Rank> out_end)
{
    return packaged_generate<FunctorType, InputValueType, OutIteratorType<OutputValueType, Rank>, Rank, Fused>(functor, out_beg, out_end);
}

template <typename FunctorType, typename OutputValueType, int Rank>
class partially_packaged_generate
{
public:
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::generate;
    static constexpr auto requirement = stage_requirement::output;

    using functor_type = FunctorType;
    using output_value_type = OutputValueType;

    partially_packaged_generate(functor_type functor, cl::sycl::range<rank> range)
        : functor_(functor), range_(range)
    {
    }

    template<typename OutIteratorType>
    auto complete(OutIteratorType beg, OutIteratorType end) const 
    { 
        return package_generate<output_value_type>(functor_, beg, end);
    }

    cl::sycl::range<rank> get_range();

private:
    functor_type functor_;
    cl::sycl::range<rank> range_;
};

template <typename OutputValueType, typename FunctorType, int Rank>
auto package_generate(FunctorType functor, cl::sycl::range<Rank> range)
{
    return partially_packaged_generate<FunctorType, OutputValueType, Rank>(functor, range);
}

namespace detail
{

template <typename FunctorType, typename InputValueType, typename OutputIteratorType, int Rank, bool Fused>
struct is_packaged_task<packaged_generate<FunctorType, InputValueType, OutputIteratorType, Rank, Fused>>
    : std::bool_constant<true>
{
};

template <typename FunctorType, typename InputValueType, int Rank>
struct is_partially_packaged_task<partially_packaged_generate<FunctorType, InputValueType, Rank>>
    : std::bool_constant<true>
{
};

} // namespace detail

} // namespace celerity::algorithm

#endif // PACKAGED_GENERATE_H