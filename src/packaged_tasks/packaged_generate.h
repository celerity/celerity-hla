#ifndef PACKAGED_GENERATE_H
#define PACKAGED_GENERATE_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../packaged_task_traits.h"

namespace celerity::algorithm
{

template <typename FunctorType, typename OutputValueType, typename OutputIteratorType, int Rank, bool Fused>
class packaged_generate
{
public:
    static_assert(!std::is_void_v<OutputValueType>);

    // static_assert(std::is_convertible_v<input_value_type, output_value_type>);

    packaged_generate(FunctorType functor, OutputIteratorType out_beg, OutputIteratorType out_end)
        : functor_(functor), out_beg_(out_beg), out_end_(out_end)
    {
        assert(out_beg_.get_buffer().get_id() == out_end_.get_buffer().get_id());
    }

    auto operator()(celerity::distr_queue &queue) const
    {
        std::invoke(get_task(), queue, out_beg_, out_end_);
        return out_beg_.get_buffer();
    }

    OutputIteratorType get_out_beg() const { return out_beg_; }
    OutputIteratorType get_out_end() const { return out_end_; }

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
    FunctorType functor_;
    OutputIteratorType out_beg_;
    OutputIteratorType out_end_;
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
    static_assert(!std::is_void_v<OutputValueType>);
 
    partially_packaged_generate(FunctorType functor, cl::sycl::range<Rank> range)
        : functor_(functor), range_(range)
    {
    }

    template<typename OutIteratorType>
    auto complete(OutIteratorType beg, OutIteratorType end) const 
    { 
        return package_generate<OutputValueType>(functor_, beg, end);
    }

    cl::sycl::range<Rank> get_range() { return range_; }

private:
    FunctorType functor_;
    cl::sycl::range<Rank> range_;
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
    : std::true_type
{
};

template <typename FunctorType, typename InputValueType, int Rank>
struct is_partially_packaged_task<partially_packaged_generate<FunctorType, InputValueType, Rank>>
    : std::true_type
{
};

template <typename FunctorType, typename OutputValueType, typename OutputIteratorType, int Rank, bool Fused>
struct packaged_task_traits<packaged_generate<FunctorType, OutputValueType, OutputIteratorType, Rank, Fused>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::generate;
    static constexpr auto access_type = access_type::invalid;

    using input_value_type = void;
    using input_iterator_type = void;
    using output_value_type = OutputValueType;
    using output_iterator_type = OutputIteratorType;
};

template <typename FunctorType, typename OutputValueType, int Rank>
struct packaged_task_traits<partially_packaged_generate<FunctorType, OutputValueType, Rank>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::generate;
    static constexpr auto access_type = access_type::invalid;

    using input_value_type = void;
    using input_iterator_type = void;
    using output_value_type = OutputValueType;
    using output_iterator_type = void;  
};

template <typename FunctorType, typename OutputValueType, int Rank>
struct partially_packaged_task_traits<partially_packaged_generate<FunctorType, OutputValueType, Rank>> 
    : packaged_task_traits<partially_packaged_generate<FunctorType, OutputValueType, Rank>>
{
    static constexpr auto requirement = stage_requirement::output; 
};

} // namespace detail

} // namespace celerity::algorithm

#endif // PACKAGED_GENERATE_H