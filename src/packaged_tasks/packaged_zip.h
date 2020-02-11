#ifndef ZIP_DECORATOR_H
#define ZIP_DECORATOR_H

#include "../iterator.h"
#include "../celerity_helper.h"
#include "../accessor_type.h"
#include "../computation_type.h"
#include "../packaged_task_traits.h"

namespace celerity::algorithm
{

template <typename FunctorType, 
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          typename OutputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          bool Fused>
class packaged_zip
{
public:
    packaged_zip(FunctorType functor,
                 FirstInputIteratorType in_beg,
                 FirstInputIteratorType in_end,
                 SecondInputIteratorType second_in_beg,
                 OutputIteratorType out_beg)
        : functor_(functor), in_beg_(in_beg), in_end_(in_end), second_in_beg_(second_in_beg), out_beg_(out_beg)
    {
        assert(are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
        assert(!are_equal(in_beg_.get_buffer(), out_beg_.get_buffer()));
        assert(!are_equal(second_in_beg_.get_buffer(), out_beg_.get_buffer()));
    }

    void operator()(celerity::distr_queue &queue) const
    {
        std::invoke(get_task(), queue, in_beg_, in_end_);
    }

    auto get_task() 
    {
        if constexpr (Fused)
        {
            return functor_;
        }
        else
        {
            return functor_(in_beg_, in_end_, second_in_beg_, out_beg_);
        }
    }

    FirstInputIteratorType get_in_beg() const { return in_beg_; }
    FirstInputIteratorType get_in_end() const { return in_end_; }
    SecondInputIteratorType get_second_in_beg() const { return second_in_beg_; }
    OutputIteratorType get_out_iterator() const { return out_beg_; }

    cl::sycl::range<Rank> get_range() const { return distance(in_beg_, in_end_); }

private:
    FunctorType functor_;
    FirstInputIteratorType in_beg_;
    FirstInputIteratorType in_end_;
    SecondInputIteratorType second_in_beg_;
    OutputIteratorType out_beg_;
};

template <access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          typename FunctorType, 
          template<typename, int> typename InIteratorType, 
          template<typename, int> typename SecondInIteratorType, 
          template<typename, int> typename OutIteratorType, 
          typename FirstInputValueType, 
          typename SecondInputValueType, 
          typename OutputValueType, 
          int Rank>
auto package_zip(FunctorType task,
                 InIteratorType<FirstInputValueType, Rank> in_beg,
                 InIteratorType<FirstInputValueType, Rank> in_end,
                 SecondInIteratorType<SecondInputValueType, Rank> second_in_beg,
                 OutIteratorType<OutputValueType, Rank> out_beg)
{
    return packaged_zip<FunctorType, 
                        InIteratorType<FirstInputValueType, Rank>, 
                        SecondInIteratorType<SecondInputValueType, Rank>, 
                        OutIteratorType<OutputValueType, Rank>, 
                        Rank, 
                        FirstInputAccessType, 
                        SecondInputAccessType,
                        false>(task, in_beg, in_end, second_in_beg, out_beg);
}

template <access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          bool Fused,
          typename FunctorType, 
          template<typename, int> typename InIteratorType, 
          template<typename, int> typename SecondInIteratorType, 
          template<typename, int> typename OutIteratorType, 
          typename FirstInputValueType, 
          typename SecondInputValueType, 
          typename OutputValueType, 
          int Rank>
auto package_zip(FunctorType task,
                 InIteratorType<FirstInputValueType, Rank> in_beg,
                 InIteratorType<FirstInputValueType, Rank> in_end,
                 SecondInIteratorType<SecondInputValueType, Rank> second_in_beg,
                 OutIteratorType<OutputValueType, Rank> out_beg)
{
    return packaged_zip<FunctorType, 
                        InIteratorType<FirstInputValueType, Rank>, 
                        SecondInIteratorType<SecondInputValueType, Rank>, 
                        OutIteratorType<OutputValueType, Rank>, 
                        Rank, 
                        FirstInputAccessType, 
                        SecondInputAccessType,
                        Fused>(task, in_beg, in_end, second_in_beg, out_beg);
}

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
class partially_packaged_zip_2
{
public:
    partially_packaged_zip_2(FunctorType functor,
                             FirstInputIteratorType in_beg,
                             FirstInputIteratorType in_end,
                             SecondInputIteratorType second_in_beg)
        : functor_(functor), in_beg_(in_beg), in_end_(in_end), second_in_beg_(second_in_beg)
    {
        assert(are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
        assert(!are_equal(in_beg_.get_buffer(), second_in_beg_.get_buffer()));
    }

    template<typename Iterator>
    auto complete(Iterator beg, Iterator)
    {
        return package_zip<FirstInputAccessType, SecondInputAccessType>(
            functor_, in_beg_, in_end_, second_in_beg_, beg);
    }

    FirstInputIteratorType get_in_beg() const { return in_beg_; }
    FirstInputIteratorType get_in_end() const { return in_end_; }
    SecondInputIteratorType get_second_in_beg() const { return second_in_beg_; }

    cl::sycl::range<Rank> get_range() const { return distance(in_beg_, in_end_); }

private:
    FunctorType functor_;
    FirstInputIteratorType in_beg_;
    FirstInputIteratorType in_end_;
    SecondInputIteratorType second_in_beg_;
};

template <access_type FirstInputAccessType,
          access_type SecondInputAccessType,
          typename KernelType,
          typename FunctorType, 
          template<typename, int> typename InIteratorType, 
          template<typename, int> typename SecondInIteratorType, 
          typename FirstInputValueType, 
          typename SecondInputValueType, 
          int Rank>
auto package_zip(FunctorType functor,
                 InIteratorType<FirstInputValueType, Rank> in_beg,
                 InIteratorType<FirstInputValueType, Rank> in_end,
                 SecondInIteratorType<SecondInputValueType, Rank> second_in_beg)
{
    return partially_packaged_zip_2<FunctorType, 
                                    KernelType,
                                    InIteratorType<FirstInputValueType, Rank>, 
                                    SecondInIteratorType<SecondInputValueType, Rank>,  
                                    Rank, 
                                    FirstInputAccessType, 
                                    SecondInputAccessType>(functor, in_beg, in_end, second_in_beg);
}

// Rename FirstInputIterator to SecondInputIterator 
// we are switching those in this stage
template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType,
          access_type SecondInputAccessType>
class partially_packaged_zip_1
{
public:
    partially_packaged_zip_1(FunctorType functor,
                             FirstInputIteratorType in_beg,
                             FirstInputIteratorType in_end)
        : functor_(functor), in_beg_(in_beg), in_end_(in_end)
    {
        assert(are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
    }

    template<typename Iterator>
    auto complete(Iterator beg, Iterator end)
    {
        return package_zip<FirstInputAccessType, SecondInputAccessType, KernelType>(
            functor_, beg, end, in_beg_);
    }

    FirstInputIteratorType get_in_beg() const { return in_beg_; }
    FirstInputIteratorType get_in_end() const { return in_end_; }

    cl::sycl::range<Rank> get_range() const { return distance(in_beg_, in_end_); }

private:
    FunctorType functor_;
    FirstInputIteratorType in_beg_;
    FirstInputIteratorType in_end_;
};

template <access_type FirstInputAccessType,
          access_type SecondInputAccessType,
          typename KernelType,
          typename FunctorType, 
          template<typename, int> typename InIteratorType, 
          typename FirstInputValueType,
          int Rank>
auto package_zip(FunctorType functor,
                 InIteratorType<FirstInputValueType, Rank> in_beg,
                 InIteratorType<FirstInputValueType, Rank> in_end)
{
    return partially_packaged_zip_1<FunctorType, 
                                    KernelType,
                                    InIteratorType<FirstInputValueType, Rank>,
                                    Rank, 
                                    FirstInputAccessType, 
                                    SecondInputAccessType>(functor, in_beg, in_end);
}

template <typename FunctorType, 
          typename KernelType,
          access_type FirstInputAccessType,
          access_type SecondInputAccessType>
class partially_packaged_zip_0
{
public:
    partially_packaged_zip_0(FunctorType functor)
        : functor_(functor)
    {

    }

    template<typename Iterator>
    auto complete(Iterator beg, Iterator end)
    {
        return package_zip<FirstInputAccessType, SecondInputAccessType, KernelType>(
            functor_, beg, end);
    }

private:
    FunctorType functor_;
};

template <access_type FirstInputAccessType,
          access_type SecondInputAccessType,
          typename KernelType,
          typename FunctorType>
auto package_zip(FunctorType functor)
{
    return partially_packaged_zip_0<FunctorType, 
                                    KernelType,
                                    FirstInputAccessType, 
                                    SecondInputAccessType>(functor);
}

namespace detail
{


template <typename FunctorType, 
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          typename OutputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          bool Fused>
struct is_packaged_task<packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType, Fused>>
    : std::bool_constant<true>
{
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct is_partially_packaged_task<partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
    : std::bool_constant<true>
{
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct is_partially_packaged_task<partially_packaged_zip_1<FunctorType, KernelType, FirstInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
    : std::bool_constant<true>
{
};


template <typename FunctorType, 
          typename KernelType,
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct is_partially_packaged_task<partially_packaged_zip_0<FunctorType, KernelType, FirstInputAccessType, SecondInputAccessType>>
    : std::bool_constant<true>
{
};

template <typename FunctorType, 
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          typename OutputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          bool Fused>
struct packaged_task_traits<packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType, Fused>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::zip;
    static constexpr auto access_type = FirstInputAccessType;

    using input_value_type = typename std::iterator_traits<FirstInputIteratorType>::value_type;
    using output_value_type = typename std::iterator_traits<OutputIteratorType>::value_type;

    using input_iterator_type = FirstInputIteratorType;
    using output_iterator_type = OutputIteratorType;
};

template <typename FunctorType, 
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          typename OutputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType, 
          bool Fused>
struct extended_packaged_task_traits<packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType, Fused>, computation_type::zip>
{
    static constexpr auto second_input_access_type = SecondInputAccessType;

    using second_input_value_type = typename std::iterator_traits<SecondInputIteratorType>::value_type;
    using second_input_iterator_type = SecondInputIteratorType;
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct packaged_task_traits<partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::zip;
    static constexpr auto access_type = FirstInputAccessType;

    using input_value_type = typename std::iterator_traits<FirstInputIteratorType>::value_type;
    using output_value_type = kernel_result_t<KernelType>;

    using input_iterator_type = FirstInputIteratorType;
    using output_iterator_type = void;
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct extended_packaged_task_traits<partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>, computation_type::zip>
{
    static constexpr auto second_input_access_type = SecondInputAccessType;

    using second_input_value_type = typename std::iterator_traits<SecondInputIteratorType>::value_type;
    using second_input_iterator_type = SecondInputIteratorType;
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct packaged_task_traits<partially_packaged_zip_1<FunctorType, KernelType, FirstInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto rank = Rank;
    static constexpr auto computation_type = computation_type::zip;
    static constexpr auto access_type = FirstInputAccessType;

    using input_value_type = typename std::iterator_traits<FirstInputIteratorType>::value_type;
    using output_value_type = kernel_result_t<KernelType>;

    using input_iterator_type = FirstInputIteratorType;
    using output_iterator_type = void;
};


template <typename FunctorType, 
          typename KernelType,
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct packaged_task_traits<partially_packaged_zip_0<FunctorType, KernelType, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto computation_type = computation_type::zip;
    static constexpr auto access_type = FirstInputAccessType;

    using input_value_type = void;
    using output_value_type = kernel_result_t<KernelType>;

    using input_iterator_type = void;
    using output_iterator_type = void;
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          typename SecondInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct partially_packaged_task_traits<partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto requirement = stage_requirement::output;
};

template <typename FunctorType, 
          typename KernelType,
          typename FirstInputIteratorType, 
          int Rank, 
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct partially_packaged_task_traits<partially_packaged_zip_1<FunctorType, KernelType, FirstInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto requirement = stage_requirement::input;
};


template <typename FunctorType, 
          typename KernelType,
          access_type FirstInputAccessType, 
          access_type SecondInputAccessType>
struct partially_packaged_task_traits<partially_packaged_zip_0<FunctorType, KernelType, FirstInputAccessType, SecondInputAccessType>>
{
    static constexpr auto requirement = stage_requirement::input;
};

} // namespace detail

} // namespace celerity::algorithm

#endif // ZIP_DECORATOR_H