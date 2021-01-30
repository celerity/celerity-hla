#ifndef CELERITY_HLA_PACKAGED_TRANSFORM_H
#define CELERITY_HLA_PACKAGED_TRANSFORM_H

#include "../../iterator.h"
#include "../../celerity_helper.h"
#include "../../accessor_type.h"
#include "../../computation_type.h"
#include "../../packaged_task_traits.h"
#include "../../partially_packaged_task.h"
#include "../../accessor_type.h"

// experimental
#include "../kernel_traits.h"

namespace celerity::hla::experimental::detail
{

    template <int Rank, celerity::algorithm::detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
    class packaged_transform
    {
    public:
        static_assert(!std::is_void_v<typename std::iterator_traits<InputIteratorType>::value_type>);
        static_assert(!std::is_void_v<typename std::iterator_traits<OutputIteratorType>::value_type>);

        packaged_transform(Functor functor, InputIteratorType in_beg, InputIteratorType in_end, OutputIteratorType out_beg)
            : functor_(functor), in_beg_(in_beg), in_end_(in_end), out_beg_(out_beg)
        {
            assert(algorithm::detail::are_equal(in_beg_.get_buffer(), in_end_.get_buffer()));
            assert(InputAccessType == celerity::algorithm::detail::access_type::one_to_one || !algorithm::detail::are_equal(in_beg_.get_buffer(), out_beg_.get_buffer()));
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
        OutputIteratorType get_out_beg() const { return out_beg_; }

    private:
        Functor functor_;
        InputIteratorType in_beg_;
        InputIteratorType in_end_;
        OutputIteratorType out_beg_;
    };

    template <celerity::algorithm::detail::access_type InputAccessType, typename FunctorType, template <typename, int> typename InIteratorType, template <typename, int> typename OutIteratorType, typename InputValueType, typename OutputValueType, int Rank>
    auto package_transform(FunctorType task,
                           InIteratorType<InputValueType, Rank> in_beg,
                           InIteratorType<InputValueType, Rank> in_end,
                           OutIteratorType<OutputValueType, Rank> out_beg)
    {
        return packaged_transform<Rank, InputAccessType, FunctorType, InIteratorType<InputValueType, Rank>, OutIteratorType<OutputValueType, Rank>>(
            task, in_beg, in_end, out_beg);
    }

    template <int Rank, celerity::algorithm::detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputIteratorType>
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
            return experimental::detail::package_transform<InputAccessType>(f, in_beg_, in_end_, beg);
        }

        InputIteratorType get_in_beg() const { return in_beg_; }
        InputIteratorType get_in_end() const { return in_end_; }

        cl::sycl::range<Rank> get_range() const { return distance(in_beg_, in_end_); }

    private:
        Functor f_;
        InputIteratorType in_beg_;
        InputIteratorType in_end_;
    };

    template <celerity::algorithm::detail::access_type InputAccessType, typename KernelFunctor, typename FunctorType, int Rank, template <typename, int> typename InIteratorType, typename InputValueType>
    auto package_transform(FunctorType functor, InIteratorType<InputValueType, Rank> beg, InIteratorType<InputValueType, Rank> end)
    {
        return partially_packaged_transform_1<Rank, InputAccessType, FunctorType, KernelFunctor, InIteratorType<InputValueType, Rank>>(functor, beg, end);
    }

    template <typename Functor, typename KernelFunctor>
    class partially_packaged_transform_0
    {
    public:
        explicit partially_packaged_transform_0(Functor f)
            : f_(f) {}

        template <typename Iterator>
        auto complete(Iterator beg, Iterator end) const
        {
            constexpr auto access_type = get_access_concept<KernelFunctor, 1, 0, typename Iterator::value_type, Iterator::rank>();
            return package_transform<access_type, KernelFunctor>(f_, beg, end);
        }

    private:
        Functor f_;
    };

    template <typename KernelFunctor, typename FunctorType>
    auto package_transform(FunctorType functor)
    {
        return partially_packaged_transform_0<FunctorType, KernelFunctor>(functor);
    }
} // namespace celerity::hla::experimental::detail

namespace celerity::algorithm::traits
{

    template <int Rank, detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
    struct is_packaged_task<celerity::hla::experimental::detail::packaged_transform<Rank, InputAccessType, Functor, InputIteratorType, OutputIteratorType>>
        : std::true_type
    {
    };

    template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputValueType>
    struct is_partially_packaged_task<celerity::hla::experimental::detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputValueType>>
        : std::true_type
    {
    };

    template <typename Functor, typename KernelFunctor>
    struct is_partially_packaged_task<celerity::hla::experimental::detail::partially_packaged_transform_0<Functor, KernelFunctor>>
        : std::true_type
    {
    };

    template <int Rank, detail::access_type InputAccessType, typename Functor, typename InputIteratorType, typename OutputIteratorType>
    struct packaged_task_traits<celerity::hla::experimental::detail::packaged_transform<Rank, InputAccessType, Functor, InputIteratorType, OutputIteratorType>>
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
    struct packaged_task_traits<celerity::hla::experimental::detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputIteratorType>>
    {
        static constexpr auto rank = Rank;
        static constexpr auto computation_type = detail::computation_type::transform;
        static constexpr auto access_type = InputAccessType;

        using input_iterator_type = InputIteratorType;
        using input_value_type = typename std::iterator_traits<InputIteratorType>::value_type;
        using output_value_type = hla::experimental::kernel_result_t<KernelFunctor, std::integral_constant<int, rank>, input_value_type>;
        using output_iterator_type = void;
    };

    template <int Rank, detail::access_type InputAccessType, typename Functor, typename KernelFunctor, typename InputIteratorType>
    struct partially_packaged_task_traits<celerity::hla::experimental::detail::partially_packaged_transform_1<Rank, InputAccessType, Functor, KernelFunctor, InputIteratorType>>
    {
        static constexpr auto requirement = detail::stage_requirement::output;
    };

    template <typename Functor, typename KernelFunctor>
    struct packaged_task_traits<celerity::hla::experimental::detail::partially_packaged_transform_0<Functor, KernelFunctor>>
    {
        static constexpr auto rank = -1;
        static constexpr auto computation_type = detail::computation_type::transform;

        template <typename Input>
        static constexpr auto access_type = hla::experimental::access_concept_v<KernelFunctor, 0, 1,
                                                                                typename packaged_task_traits<Input>::output_value_type,
                                                                                packaged_task_traits<Input>::rank>;

        using input_iterator_type = void;
        using input_value_type = void;
        using output_value_type = void;
        using output_iterator_type = void;
    };

    template <typename Functor, typename KernelFunctor>
    struct partially_packaged_task_traits<celerity::hla::experimental::detail::partially_packaged_transform_0<Functor, KernelFunctor>>
    {
        static constexpr auto requirement = detail::stage_requirement::input;
    };

} // namespace celerity::algorithm::traits

#endif // CELERITY_HLA_PACKAGED_TRANSFORM_H