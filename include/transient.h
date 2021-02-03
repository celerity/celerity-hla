#ifndef TRANSIENT_H
#define TRANSIENT_H

#include "platform.h"
#include "celerity_helper.h"
#include "packaged_task_traits.h"
#include "item_context.h"
#include "sequence.h"
#include "transient_traits.h"
#include "experimental/traits.h"

namespace celerity::algorithm
{

    namespace detail
    {

        template <typename T, int Rank, cl::sycl::access::mode Mode>
        struct transient_accessor
        {
        public:
            transient_accessor() {}

            auto operator[](item_shared_data<Rank, T> ctx) const -> std::conditional_t<Mode == cl::sycl::access::mode::read, T, T &>
            {
                return ctx.get();
            }

            auto operator[](cl::sycl::item<Rank>) const -> std::conditional_t<Mode == cl::sycl::access::mode::read, T, T &>
            {
                on_error("transient_accessor", "can only operate on item_shared_data");
            }
        };

        template <typename T, int Rank>
        struct transient_buffer
        {
        public:
            static_assert(!std::is_void_v<T>);

            static unsigned long long curr_id;

            explicit transient_buffer(cl::sycl::range<Rank> range)
                : range_(range), id_(curr_id++) {}

            template <cl::sycl::access::mode Mode, typename RangeMapper>
            auto get_access(handler &cgh, RangeMapper)
            {
                return transient_accessor<T, Rank, Mode>{};
            }

            cl::sycl::range<Rank> get_range() const { return range_; }
            unsigned long long get_id() const { return id_; }

        private:
            cl::sycl::range<Rank> range_;
            unsigned long long id_;
        };

        template <typename T, int Rank>
        unsigned long long transient_buffer<T, Rank>::curr_id = 0;

        template <typename T, int Rank>
        struct transient_iterator : iterator<Rank>
        {
        public:
            using iterator_category = celerity_iterator_tag;
            using value_type = T;
            using difference_type = long;
            using pointer = std::add_pointer_t<T>;
            using reference = std::add_lvalue_reference_t<T>;

            static constexpr auto rank = Rank;

            transient_iterator(cl::sycl::id<Rank> pos, transient_buffer<T, Rank> buffer)
                : iterator<Rank>(pos, buffer.get_range()), buffer_(buffer) {}

            [[nodiscard]] transient_buffer<T, Rank> get_buffer() const
            {
                return buffer_;
            }

        private:
            transient_buffer<T, Rank> buffer_;
        };

        template <typename ElementTypeA, int RankA,
                  typename ElementTypeB, int RankB>
        bool are_equal(buffer<ElementTypeA, RankA> a, transient_buffer<ElementTypeB, RankB> b)
        {
            return false;
        }

        template <typename ElementTypeA, int RankA,
                  typename ElementTypeB, int RankB>
        bool are_equal(transient_buffer<ElementTypeA, RankA> a, buffer<ElementTypeB, RankB> b)
        {
            return false;
        }

        template <typename ElementType, int Rank>
        bool are_equal(transient_buffer<ElementType, Rank> a, transient_buffer<ElementType, Rank> b)
        {
            return a.get_id() == b.get_id();
        }

        template <typename T, int Rank>
        transient_iterator<T, Rank> begin(transient_buffer<T, Rank> buffer)
        {
            return transient_iterator<T, Rank>(cl::sycl::id<Rank>{}, buffer);
        }

        template <typename T, int Rank>
        transient_iterator<T, Rank> end(transient_buffer<T, Rank> buffer)
        {
            return transient_iterator<T, Rank>(buffer.get_range(), buffer);
        }

    } // namespace detail

    namespace traits
    {
        template <typename T, int Rank>
        struct is_transient<detail::transient_iterator<T, Rank>> : std::true_type
        {
        };
    } // namespace traits

} // namespace celerity::algorithm

namespace celerity::hla::experimental
{
    template <typename ValueType, size_t Rank>
    struct is_kernel_input<algorithm::detail::transient_iterator<ValueType, Rank>> : std::bool_constant<true>
    {
    };
} // namespace celerity::hla::experimental

#endif // TRANSIENT_H