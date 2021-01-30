#ifndef CELERITY_HLA_ZIP_DECORATOR_H
#define CELERITY_HLA_ZIP_DECORATOR_H

#include "../../iterator.h"
#include "../../celerity_helper.h"
#include "../../accessor_type.h"
#include "../../computation_type.h"
#include "../../packaged_task_traits.h"
#include "../../partially_packaged_task.h"
namespace celerity::hla::experimental::detail
{
    template <typename FunctorType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              typename OutputIteratorType,
              int Rank,
              algorithm::detail::access_type FirstInputAccessType,
              algorithm::detail::access_type SecondInputAccessType>
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
            assert(algorithm::detail::are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
            assert(FirstInputAccessType == algorithm::detail::access_type::one_to_one || !algorithm::detail::are_equal(in_beg_.get_buffer(), out_beg_.get_buffer()));
            assert(SecondInputAccessType == algorithm::detail::access_type::one_to_one || !algorithm::detail::are_equal(second_in_beg_.get_buffer(), out_beg_.get_buffer()));
            //assert(distance(in_beg_, in_end_) <= distance(out_beg_, end(out_beg_.get_buffer())));
        }

        auto operator()(celerity::distr_queue &queue) const
        {
            std::invoke(get_task(), queue, in_beg_, in_end_);
            return out_beg_.get_buffer();
        }

        auto get_task() const { return functor_; }

        FirstInputIteratorType get_in_beg() const { return in_beg_; }
        FirstInputIteratorType get_in_end() const { return in_end_; }
        SecondInputIteratorType get_second_in_beg() const { return second_in_beg_; }
        OutputIteratorType get_out_beg() const { return out_beg_; }

        cl::sycl::range<Rank> get_range() const { return algorithm::detail::distance(in_beg_, in_end_); }

    private:
        FunctorType functor_;
        FirstInputIteratorType in_beg_;
        FirstInputIteratorType in_end_;
        SecondInputIteratorType second_in_beg_;
        OutputIteratorType out_beg_;
    };

    template <algorithm::detail::access_type FirstInputAccessType,
              algorithm::detail::access_type SecondInputAccessType,
              typename FunctorType,
              template <typename, int> typename InIteratorType,
              template <typename, int> typename SecondInIteratorType,
              template <typename, int> typename OutIteratorType,
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
                            SecondInputAccessType>(task, in_beg, in_end, second_in_beg, out_beg);
    }

    template <typename FunctorType,
              typename KernelType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              int Rank,
              algorithm::detail::access_type FirstInputAccessType,
              algorithm::detail::access_type SecondInputAccessType>
    class partially_packaged_zip_2
    {
    public:
        partially_packaged_zip_2(FunctorType functor,
                                 FirstInputIteratorType in_beg,
                                 FirstInputIteratorType in_end,
                                 SecondInputIteratorType second_in_beg)
            : functor_(functor), in_beg_(in_beg), in_end_(in_end), second_in_beg_(second_in_beg)
        {
            assert(algorithm::detail::are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
            //assert(!algorithm::detail::are_equal(in_beg_.get_buffer(), second_in_beg_.get_buffer()));
        }

        template <typename Iterator>
        auto complete(Iterator beg, Iterator)
        {
            const auto f = std::invoke(functor_, in_beg_, in_end_, second_in_beg_, beg);
            return hla::experimental::detail::package_zip<FirstInputAccessType, SecondInputAccessType>(
                f, in_beg_, in_end_, second_in_beg_, beg);
        }

        FirstInputIteratorType get_in_beg() const { return in_beg_; }
        FirstInputIteratorType get_in_end() const { return in_end_; }
        SecondInputIteratorType get_second_in_beg() const { return second_in_beg_; }

        cl::sycl::range<Rank> get_range() const { return algorithm::detail::distance(in_beg_, in_end_); }

    private:
        FunctorType functor_;
        FirstInputIteratorType in_beg_;
        FirstInputIteratorType in_end_;
        SecondInputIteratorType second_in_beg_;
    };

    template <algorithm::detail::access_type FirstInputAccessType,
              algorithm::detail::access_type SecondInputAccessType,
              typename KernelType,
              typename FunctorType,
              template <typename, int> typename InIteratorType,
              template <typename, int> typename SecondInIteratorType,
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

    template <typename FunctorType,
              typename KernelType,
              typename SecondInputIteratorType,
              int Rank,
              algorithm::detail::access_type FirstInputAccessType>
    class partially_packaged_zip_1
    {
    public:
        partially_packaged_zip_1(FunctorType functor,
                                 SecondInputIteratorType in_beg,
                                 SecondInputIteratorType in_end)
            : functor_(functor), in_beg_(in_beg), in_end_(in_end)
        {
            assert(algorithm::detail::are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
        }

        template <typename Iterator>
        auto complete(Iterator beg, Iterator end)
        {
            constexpr auto second_access_type = get_access_concept<KernelType, 2, 1, typename Iterator::value_type, Iterator::rank>();

            return hla::experimental::detail::package_zip<FirstInputAccessType, second_access_type, KernelType>(
                functor_, beg, end, in_beg_);
        }

        SecondInputIteratorType get_in_beg() const { return in_beg_; }
        SecondInputIteratorType get_in_end() const { return in_end_; }

        cl::sycl::range<Rank> get_range() const { return algorithm::detail::distance(in_beg_, in_end_); }

    private:
        FunctorType functor_;
        SecondInputIteratorType in_beg_;
        SecondInputIteratorType in_end_;
    };

    template <algorithm::detail::access_type FirstInputAccessType,
              typename KernelType,
              typename FunctorType,
              template <typename, int> typename InIteratorType,
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
                                        FirstInputAccessType>(functor, in_beg, in_end);
    }

    template <typename FunctorType,
              typename KernelType>
    class partially_packaged_zip_0
    {
    public:
        partially_packaged_zip_0(FunctorType functor)
            : functor_(functor)
        {
        }

        template <typename Iterator>
        auto complete(Iterator beg, Iterator end)
        {
            constexpr auto first_access_type = get_access_concept<KernelType, 2, 0, typename Iterator::value_type, Iterator::rank>();
            return package_zip<first_access_type, KernelType>(
                functor_, beg, end);
        }

    private:
        FunctorType functor_;
    };

    template <typename KernelType,
              typename FunctorType>
    auto package_zip(FunctorType functor)
    {
        return partially_packaged_zip_0<FunctorType,
                                        KernelType>(functor);
    }
} // namespace celerity::hla::experimental::detail

namespace celerity::algorithm::traits
{

    template <typename FunctorType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              typename OutputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct is_packaged_task<hla::experimental::detail::packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
        : std::bool_constant<true>
    {
    };

    template <typename FunctorType,
              typename KernelType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct is_partially_packaged_task<hla::experimental::detail::partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
        : std::bool_constant<true>
    {
    };

    template <typename FunctorType,
              typename KernelType,
              typename SecondInputIterator,
              int Rank,
              detail::access_type FirstInputAccessType>
    struct is_partially_packaged_task<hla::experimental::detail::partially_packaged_zip_1<FunctorType, KernelType, SecondInputIterator, Rank, FirstInputAccessType>>
        : std::bool_constant<true>
    {
    };

    template <typename FunctorType,
              typename KernelType>
    struct is_partially_packaged_task<hla::experimental::detail::partially_packaged_zip_0<FunctorType, KernelType>>
        : std::bool_constant<true>
    {
    };

    template <typename FunctorType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              typename OutputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct packaged_task_traits<hla::experimental::detail::packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
    {
        static constexpr auto rank = Rank;
        static constexpr auto computation_type = detail::computation_type::zip;

        template <typename>
        static constexpr detail::access_type access_type = FirstInputAccessType;

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
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct extended_packaged_task_traits<hla::experimental::detail::packaged_zip<FunctorType, FirstInputIteratorType, SecondInputIteratorType, OutputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>, detail::computation_type::zip>
    {
        template <typename>
        static constexpr detail::access_type second_input_access_type = SecondInputAccessType;

        using second_input_value_type = typename std::iterator_traits<SecondInputIteratorType>::value_type;
        using second_input_iterator_type = SecondInputIteratorType;
    };

    template <typename FunctorType,
              typename KernelType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct packaged_task_traits<hla::experimental::detail::partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
    {
        static constexpr auto rank = Rank;
        static constexpr auto computation_type = detail::computation_type::zip;

        template <typename>
        static constexpr detail::access_type access_type = FirstInputAccessType;

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
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct extended_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>, detail::computation_type::zip>
    {
        template <typename>
        static constexpr detail::access_type second_input_access_type = SecondInputAccessType;

        using second_input_value_type = typename std::iterator_traits<SecondInputIteratorType>::value_type;
        using second_input_iterator_type = SecondInputIteratorType;
    };

    template <typename FunctorType,
              typename KernelType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType>
    struct packaged_task_traits<hla::experimental::detail::partially_packaged_zip_1<FunctorType, KernelType, SecondInputIteratorType, Rank, FirstInputAccessType>>
    {
        static constexpr auto rank = Rank;
        static constexpr auto computation_type = detail::computation_type::zip;

        template <typename>
        static constexpr detail::access_type access_type = FirstInputAccessType;

        using input_value_type = void;
        using output_value_type = kernel_result_t<KernelType>;

        using input_iterator_type = void;
        using output_iterator_type = void;
    };

    template <typename FunctorType,
              typename KernelType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType>
    struct extended_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_1<FunctorType, KernelType, SecondInputIteratorType, Rank, FirstInputAccessType>, detail::computation_type::zip>
    {
        template <typename Input>
        static constexpr detail::access_type second_input_access_type = hla::experimental::access_concept_v<KernelType, 2, 1,
                                                                                                            typename packaged_task_traits<Input>::output_value_type,
                                                                                                            packaged_task_traits<Input>::rank>;

        using second_input_value_type = typename std::iterator_traits<SecondInputIteratorType>::value_type;
        using second_input_iterator_type = SecondInputIteratorType;
    };

    template <typename FunctorType,
              typename KernelType>
    struct packaged_task_traits<hla::experimental::detail::partially_packaged_zip_0<FunctorType, KernelType>>
    {
        static constexpr auto rank = -1;
        static constexpr auto computation_type = detail::computation_type::zip;

        template <typename Input>
        static constexpr auto access_type = hla::experimental::access_concept_v<KernelType, 2, 0,
                                                                                typename packaged_task_traits<Input>::output_value_type,
                                                                                packaged_task_traits<Input>::rank>;

        using input_value_type = void;
        using output_value_type = kernel_result_t<KernelType>;

        using input_iterator_type = void;
        using output_iterator_type = void;
    };

    template <typename FunctorType,
              typename KernelType>
    struct extended_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_0<FunctorType, KernelType>, detail::computation_type::zip>
    {
        template <typename Input>
        static constexpr auto second_input_access_type = hla::experimental::access_concept_v<KernelType, 2, 1,
                                                                                             typename packaged_task_traits<Input>::output_value_type,
                                                                                             packaged_task_traits<Input>::rank>;

        using second_input_value_type = void;
        using second_input_iterator_type = void;
    };

    template <typename FunctorType,
              typename KernelType,
              typename FirstInputIteratorType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType,
              detail::access_type SecondInputAccessType>
    struct partially_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_2<FunctorType, KernelType, FirstInputIteratorType, SecondInputIteratorType, Rank, FirstInputAccessType, SecondInputAccessType>>
    {
        static constexpr auto requirement = detail::stage_requirement::output;
    };

    template <typename FunctorType,
              typename KernelType,
              typename SecondInputIteratorType,
              int Rank,
              detail::access_type FirstInputAccessType>
    struct partially_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_1<FunctorType, KernelType, SecondInputIteratorType, Rank, FirstInputAccessType>>
    {
        static constexpr auto requirement = detail::stage_requirement::input;
    };

    template <typename FunctorType,
              typename KernelType>
    struct partially_packaged_task_traits<hla::experimental::detail::partially_packaged_zip_0<FunctorType, KernelType>>
    {
        static constexpr auto requirement = detail::stage_requirement::input;
    };

} // namespace celerity::algorithm::traits

#endif // CELERITY_HLA_ZIP_DECORATOR_H