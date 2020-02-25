#ifndef FUSION_H
#define FUSION_H

#include "celerity_helper.h"
#include "packaged_task_traits.h"
#include "item_context.h"
#include "sequence.h"


namespace celerity::algorithm
{

template<typename T>
struct is_item_context : std::bool_constant<false>
{

};

template<int Rank, typename T>
class item_shared_data
{
public:
    item_shared_data(T& data, cl::sycl::item<Rank> item) 
        : data_(data), item_(item) {}

    T& get() const { return data_; }

    operator cl::sycl::item<Rank> ()
    {
        return item_;
    }

    operator cl::sycl::id<Rank> ()
    {
        return item_.get_id();
    }

private:
    T& data_;
    cl::sycl::item<Rank> item_;
};

template<int Rank, typename ContextType>
class item_context
{
public:
    using item_type = cl::sycl::item<Rank>;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    item_shared_data<Rank, ContextType> operator[](int idx)
    {
        return { shared_data_[idx], item_ };
    }

    cl::sycl::item<Rank> get_item() const { return item_; }

private:
    cl::sycl::item<Rank> item_;
    std::array<ContextType, 2> shared_data_;
};

template<int Rank, typename ContextType>
struct is_item_context<item_context<Rank, ContextType>> : std::bool_constant<true>
{

};

// TODO: FIX THIS!
// template<int Rank, typename ContextType>
// struct is_item_context<std::tuple<item_context<Rank, ContextType>, item_context<Rank, ContextType>>> : std::bool_constant<true>
// {

// };

template<typename T>
inline constexpr bool is_item_context_v = is_item_context<T>::value;

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
        abort();
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
    auto get_access(handler& cgh, RangeMapper)
    {
        //auto acc = buffer_.template get_access<Mode>(cgh, celerity::access::fixed<Rank, 1>(celerity::subrange<1>{{0}, {1}}));
        //auto acc = buffer_.template get_access<cl::sycl::access::mode::discard_read_write, cl::sycl::access::target::local>();
        return transient_accessor<T, Rank, Mode>{ };
    }

    cl::sycl::range<Rank> get_range() const { return range_; }
    unsigned long long get_id() const { return id_; }

private:
    cl::sycl::range<Rank> range_;
    unsigned long long id_;
};

template <typename T, int Rank>
unsigned long long transient_buffer<T, Rank>::curr_id = 0;

template<typename T>
struct is_transient : std::false_type { };

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
struct is_transient<transient_iterator<T, Rank>> : std::true_type {};

template<typename T>
inline constexpr bool is_transient_v = is_transient<T>::value;

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

template<typename Task, typename SecondaryInputSequence>
struct t_joint
{
public:
    t_joint(Task task, SecondaryInputSequence sequence)
        : task_(task), secondary_in_(sequence)
    {

    }

    auto operator()(celerity::distr_queue& queue) const
    {
        std::invoke(secondary_in_, queue);
        return std::invoke(task_, queue);
    }

    auto get_in_beg() const { return task_.get_in_beg(); }
    auto get_in_end() const { return task_.get_in_end(); }
    auto get_out_iterator() const { return task_.get_out_iterator(); }
    auto get_range() const { return task_.get_range(); }

    auto get_task() { return task_; }
    auto get_secondary() { return secondary_in_; }

private:
    Task task_;
    SecondaryInputSequence secondary_in_;
};

template<typename Task, typename SecondaryInputSequence>
struct partial_t_joint
{
public:
    partial_t_joint(Task task, SecondaryInputSequence sequence)
        : task_(task), secondary_in_(sequence)
    {

    }

    template<typename IteratorType>
    auto complete(IteratorType beg, IteratorType end)
    {
        auto completed_task = task_.complete(beg, end);

        using completed_task_type = decltype(completed_task);

        if constexpr (detail::is_partially_packaged_task_v<completed_task_type>)
        {
            return partial_t_joint<completed_task_type, SecondaryInputSequence>{ 
                completed_task, secondary_in_ 
            };
        }
        else
        {
            return t_joint<completed_task_type, SecondaryInputSequence>{ 
                completed_task, secondary_in_ 
            };
        }
    }

    auto get_in_beg() const { return task_.get_in_beg(); }
    auto get_in_end() const { return task_.get_in_end(); }
    auto get_range() const { return task_.get_range(); }

private:
    Task task_;
    SecondaryInputSequence secondary_in_;
};

namespace detail
{

template<typename Task, typename SecondaryInputSequence>
struct is_packaged_task<t_joint<Task, SecondaryInputSequence>>
    : std::bool_constant<true>
{
};

template<typename Task, typename SecondaryInputSequence>
struct packaged_task_traits<t_joint<Task, SecondaryInputSequence>>
{
    using traits = packaged_task_traits<Task>;

    static constexpr auto rank = traits::rank;
    static constexpr auto computation_type = traits::computation_type;
    static constexpr auto access_type = traits::access_type;

    using input_iterator_type = typename traits::input_iterator_type;
    using input_value_type = typename traits::input_value_type;
    using output_value_type = typename traits::output_value_type;
    using output_iterator_type = typename traits::output_iterator_type;
};

template<typename Task, typename SecondaryInputSequence>
struct extended_packaged_task_traits<t_joint<Task, SecondaryInputSequence>, computation_type::zip> 
    : extended_packaged_task_traits<Task, computation_type::zip>
{};

template<typename Task, typename SecondaryInputSequence>
struct is_partially_packaged_task<partial_t_joint<Task, SecondaryInputSequence>>
    : std::bool_constant<true>
{
};

template<typename Task, typename SecondaryInputSequence>
struct packaged_task_traits<partial_t_joint<Task, SecondaryInputSequence>>
{
    using traits = packaged_task_traits<Task>;

    static constexpr auto rank = traits::rank;
    static constexpr auto computation_type = traits::computation_type;
    static constexpr auto access_type = traits::access_type;

    using input_iterator_type = typename traits::input_iterator_type;
    using input_value_type = typename traits::input_value_type;
    using output_value_type = typename traits::output_value_type;
    using output_iterator_type = typename traits::output_iterator_type;
};

template<typename Task, typename SecondaryInputSequence>
struct extended_packaged_task_traits<partial_t_joint<Task, SecondaryInputSequence>, computation_type::zip> 
    : extended_packaged_task_traits<Task, computation_type::zip>
{};

template<typename Task, typename SecondaryInputSequence>
struct partially_packaged_task_traits<partial_t_joint<Task, SecondaryInputSequence>> 
    : partially_packaged_task_traits<Task> {};

template<typename T>
struct is_t_joint : std::bool_constant<false> {};

template<typename Task, typename SecondaryInputSequence>
struct is_t_joint<t_joint<Task, SecondaryInputSequence>>

template <typename Task, typename SecondaryInputSequence>
struct is_t_joint<partial_t_joint<Task, SecondaryInputSequence>>
    : std::bool_constant<true>
{
};

template<typename T>
constexpr inline bool is_t_joint_v = is_t_joint<T>::value;

template <typename T>
struct t_joint_traits 
{
    using task_type = void;
    using secondary_input_sequence_type = sequence<>; 
};

template <typename Task, typename SecondaryInputSequence>
struct t_joint_traits<t_joint<Task, SecondaryInputSequence>>
{
    using task_type = Task;
    using secondary_input_sequence_type = SecondaryInputSequence; 
};

}

std::string to_string(access_type type)
{
    switch(type)
    {
        case access_type::one_to_one: return "one_to_one";
        case access_type::slice: return "slice";
        case access_type::chunk: return "chunk";
        case access_type::all : return "all";
        case access_type::item : return "item";
        default: return "none";
    }
}

std::string to_string(computation_type type)
{
    switch(type)
    {
        case computation_type::generate: return "generate";
        case computation_type::transform: return "transform";
        case computation_type::zip: return "zip";
        case computation_type::reduce : return "reduce";
        default: return "other";
    }
}

template<typename T>
std::string to_string()
{
    return typeid(T).name();
}

template<typename T, std::enable_if_t<detail::is_packaged_task_v<T>, int> = 0>
void to_string(std::stringstream& ss, T task)
{
    using traits = detail::packaged_task_traits<T>;

    ss << "packaged task:\n";
    ss << "  is t-joint          : " << std::boolalpha << detail::is_t_joint_v<T> << "\n";
    ss << "  type                : " << to_string(traits::computation_type) << "\n";
    ss << "  rank                : " << traits::rank << "\n";
    ss << "  access type         : " << to_string(traits::access_type) << "\n";
    ss << "  input value type    : " << to_string<typename traits::input_value_type>() << "\n";
    ss << "  output value type   : " << to_string<typename traits::output_value_type>() << "\n";
    ss << "  input iterator type : " << to_string<typename traits::input_iterator_type>() << "\n";
    ss << "  output iterator type: " << to_string<typename traits::output_iterator_type>() << "\n";

    if constexpr(traits::computation_type == computation_type::zip)
    {
        using ext_traits = detail::extended_packaged_task_traits<T, computation_type::zip>;

        ss << "\n";
        ss << "  second input access type  : " << to_string(ext_traits::second_input_access_type) << "\n";
        ss << "  second input value type   : " << to_string<typename ext_traits::second_input_value_type>() << "\n";
        ss << "  second input iterator type: " << to_string<typename ext_traits::second_input_iterator_type>() << "\n";
    }

    ss << "\n\n";
}

template<typename T, size_t...Is, std::enable_if_t<is_sequence_v<T>, int> = 0>
void to_string(std::stringstream& ss, T seq, std::index_sequence<Is...>)
{
    ((to_string(ss, std::get<Is>(seq.actions()))), ...);
}

template<typename T, std::enable_if_t<detail::is_packaged_task_sequence_v<T>, int> = 0>
std::string to_string(T seq)
{
    std::stringstream ss{};

    to_string(ss, seq, std::make_index_sequence<size_v<T>>{});

    return ss.str();
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