#ifndef FUSION_H
#define FUSION_H

#include "celerity_helper.h"

namespace celerity::algorithm
{

template<typename T>
struct is_item_context : std::bool_constant<false>
{

};

template<int Rank, typename ContextType>
class item_context
{
public:
    using item_type = cl::sycl::item<Rank>;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    operator cl::sycl::item<Rank> ()
    {
        return item_;
    }

    operator cl::sycl::id<Rank> ()
    {
        return item_.get_id();
    }

    ContextType& get() { return context_; }

private:
    cl::sycl::item<Rank> item_;
    ContextType context_;
};

template<int Rank, typename ContextType>
struct is_item_context<item_context<Rank, ContextType>> : std::bool_constant<true>
{

};

template<typename T>
inline constexpr bool is_item_context_v = is_item_context<T>::value;

template <typename T, int Rank, cl::sycl::access::mode Mode>
struct transient_accessor
{
public:
    transient_accessor() {}

    auto operator[](item_context<Rank, T>& ctx) const -> std::conditional_t<Mode == cl::sycl::access::mode::read, T, T &>
    {
        return ctx.get();
    }
};

template <typename T, int Rank>
struct transient_buffer
{
public:
    explicit transient_buffer(cl::sycl::range<Rank> range)
        : range_(range) {}

    template <cl::sycl::access::mode Mode, typename RangeMapper>
    auto get_access(handler& cgh, RangeMapper)
    {
        //auto acc = buffer_.template get_access<Mode>(cgh, celerity::access::fixed<Rank, 1>(celerity::subrange<1>{{0}, {1}}));
        //auto acc = buffer_.template get_access<cl::sycl::access::mode::discard_read_write, cl::sycl::access::target::local>();
        return transient_accessor<T, Rank, Mode>{ };
    }

    cl::sycl::range<Rank> get_range() const { return range_; }
    size_t get_id() const { return -1; }

private:
    cl::sycl::range<Rank> range_;
};

template <typename T, int Rank>
struct transient_iterator : iterator<Rank>
{
public:
    using iterator_category = celerity_iterator_tag;
    using value_type = T;
    using difference_type = long;
    using pointer = std::add_pointer_t<T>;
    using reference = std::add_lvalue_reference_t<T>;

    transient_iterator(cl::sycl::id<Rank> pos, transient_buffer<T, Rank> buffer)
        : iterator<Rank>(pos, buffer.get_range()), buffer_(buffer)
    {
    }

    [[nodiscard]] transient_buffer<T, Rank> get_buffer() const { return buffer_; }

    private : transient_buffer<T, Rank> buffer_;
};

template <typename T, int Rank>
transient_iterator<T, Rank> begin(transient_buffer<T, Rank> buffer)
{
    return algorithm::transient_iterator<T, Rank>(cl::sycl::id<Rank>{}, buffer);
}

template <typename T, int Rank>
transient_iterator<T, Rank> end(transient_buffer<T, Rank> buffer)
{
    return algorithm::transient_iterator<T, Rank>(buffer.get_range(), buffer);
}

// template <typename T>
// constexpr auto is_simple_transform_task_v = detail::computation_type_of_v<T, computation_type::transform> &&
//                                             detail::get_access_type<T>() == access_type::one_to_one;

// TODO:
//
// Does not work like that.
//
// move buffer identification to compile time using some kind of indexing
// and only do sanity checks at runtime. Needs to be done at compile time , otherwise
// we can not fuse kernels as invoking function pointers (to fused, type-erased kernels)
// is not permitted in device code.
//
// Idea is to tag buffers with ids before computation to tell them apart. Then do the same
// distinction of fusion cases as below.
//
// Another idea would be to restrain fusion to cases where there is only one input buffer and one explicit or implicit output buffer
//
//
/*template <typename T, typename U, std::enable_if_t<is_simple_transform_task_v<T> && is_simple_transform_task_v<U>, int> = 0>
auto operator|(T lhs, U rhs)
{
    // GENERAL REQUIREMENTS
    //
    // both operands are simple transformations
    //
    // CASE 1:
    //
    // both read from the same buffer and write to the same buffer
    // and access only one element each
    //
    // FUSION OPPORTUNITY:
    //
    // we can execute both transformations in sequence in the same kernel
    //

    using task_t = decltype(lhs.get_task());
    using execution_policy_t = typename task_t::execution_policy_type;

    auto f_a = lhs.get_computation_functor();
    auto f_b = rhs.get_computation_functor();

    using new_execution_policy = named_distributed_execution_policy<class hello>;

    return actions::transform(new_execution_policy{}, lhs.get_in_beg(), lhs.get_in_end(), lhs.get_out_iterator(), [](int) { return 1; });

    //return package_transform<access_type::one_to_one>([f_a, f_b](auto in_beg, auto in_end, auto out_in) {

    //auto task_a = std::invoke(f_a, in_beg, in_end, out_in);
    //auto task_b = std::invoke(f_b, in_beg, in_end, out_in);

    // using new_execution_policy = named_distributed_execution_policy<
    //     indexed_kernel_name_t<typename policy_traits<execution_policy_t>::kernel_name>>;

    //    using new_execution_policy = named_distributed_execution_policy<class hello>;

    //return task<new_execution_policy>(
    //    task_a.get_sequence() | task_b.get_sequence());
    //},
    //                                                  lhs.get_in_beg(), lhs.get_in_end(), lhs.get_out_iterator());
}
*/

} // namespace celerity::algorithm

#endif // FUSION_H