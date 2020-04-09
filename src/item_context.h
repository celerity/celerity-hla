#ifndef ITEM_CONTEXT_H
#define ITEM_CONTEXT_H

#include "celerity_helper.h"

namespace celerity::algorithm
{

namespace detail
{

template <int Rank, typename T>
class item_shared_data
{
public:
    item_shared_data(T &data, cl::sycl::item<Rank> item)
        : data_(data), item_(item) {}

    T &get() const { return data_; }

    operator cl::sycl::item<Rank>()
    {
        return item_;
    }

    operator cl::sycl::id<Rank>()
    {
        return item_.get_id();
    }

    item_shared_data &operator=(const item_shared_data &rhs)
    {
        // TODO: assert(item_ == rhs.item_);

        data_ = rhs.data_;

        return *this;
    }

private:
    T &data_;
    cl::sycl::item<Rank> item_;
};

template <int Rank, typename ContextType>
class item_context;

template <int Rank, typename OutType>
class item_context<Rank, OutType()>
{
public:
    using item_type = cl::sycl::item<Rank>;
    using in_type = void;
    using out_type = OutType;

    static constexpr auto rank = Rank;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    template <typename U>
    void copy_in(item_context<Rank, U()> &)
    {
    }

    template <typename... Us>
    void copy_out(item_context<Rank, OutType(Us...)> &other)
    {
        get_out() = other.get_out();
    }

    item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }

    cl::sycl::item<Rank> get_item() const { return item_; }

private:
    OutType out_;
    cl::sycl::item<Rank> item_;
};

template <int Rank, typename InType>
class item_context<Rank, void(InType)>
{
public:
    using item_type = cl::sycl::item<Rank>;
    using in_type = InType;
    using out_type = void;

    static constexpr auto rank = Rank;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    template <typename ItemContext>
    explicit item_context(ItemContext ctx)
        : item_(ctx.get_item())
    {
        static_assert(std::is_convertible_v<std::decay_t<decltype(get_in())>, decltype(ctx.get_out())>);
        get_in() = ctx.get_out();
    }

    template <typename ItemContextA, typename ItemContextB>
    item_context(ItemContextA a, ItemContextB b)
        : item_context(a)
    {
    }

    template <typename U>
    void copy_in(item_context<Rank, U(InType)> &other)
    {
        get_in() = other.get_in();
    }

    template <typename... Us>
    void copy_out(item_context<Rank, void(Us...)> &other)
    {
    }

    item_shared_data<Rank, InType> get_in() { return {in_, item_}; }

    cl::sycl::item<Rank> get_item() const { return item_; }

private:
    InType in_;
    cl::sycl::item<Rank> item_;
};

template <int Rank, typename OutType, typename InType>
class item_context<Rank, OutType(InType)>
{
public:
    using item_type = cl::sycl::item<Rank>;
    using in_type = InType;
    using out_type = OutType;

    static constexpr auto rank = Rank;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    template <typename ItemContext>
    explicit item_context(ItemContext ctx)
        : item_(ctx.get_item())
    {
        static_assert(std::is_convertible_v<std::decay_t<decltype(get_in())>, decltype(ctx.get_out())>);
        get_in() = ctx.get_out();
    }

    template <typename ItemContextA, typename ItemContextB>
    item_context(ItemContextA a, ItemContextB b)
        : item_context(a)
    {
    }

    template <typename U>
    void copy_in(item_context<Rank, U(InType)> &other)
    {
        get_in() = other.get_in();
    }

    template <typename... Us>
    void copy_out(item_context<Rank, OutType(Us...)> &other)
    {
        get_out() = other.get_out();
    }

    item_shared_data<Rank, InType> get_in() { return {in_, item_}; }
    item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }

    cl::sycl::item<Rank> get_item() const { return item_; }

private:
    InType in_;
    OutType out_;
    cl::sycl::item<Rank> item_;
};

template <int Rank, typename OutType, typename T, typename... Ts>
class item_context<Rank, OutType(T, Ts...)>
{
public:
    using item_type = cl::sycl::item<Rank>;

    using in_type = std::tuple<T, Ts...>;
    using out_type = OutType;

    static constexpr auto rank = Rank;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    // template<typename ItemContext>
    // explicit item_context(ItemContextA a)
    //     : item_(item)
    // {
    //     static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<0>)>, decltype(a.get_out())>);
    //     get_in<0> = a.get_out();
    // }

    template <typename ItemContextA, typename ItemContextB>
    item_context(ItemContextA a, ItemContextB b)
        : item_(a.get_item())
    {
        static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<0>())>, decltype(a.get_out())>);
        static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<1>())>, decltype(b.get_out())>);

        //assert(a.get_item() == b.get_item());

        get_in<0>() = a.get_out();
        get_in<1>() = b.get_out();
    }

    template <size_t Index>
    auto get_in()
    {
        static_assert(Index < std::tuple_size_v<decltype(in_)>);
        return item_shared_data{std::get<Index>(in_), item_};
    }

    template <typename U>
    void copy_in(item_context<Rank, U(T, Ts...)> &other)
    {
        static_assert(sizeof...(Ts) == 1);
        get_in<0>() = other.template get_in<0>();
        get_in<1>() = other.template get_in<1>();
    }

    template <typename U>
    void copy_in_rev(item_context<Rank, U(T, Ts...)> &other)
    {
        static_assert(sizeof...(Ts) == 1);
        get_in<0>() = other.template get_in<0>();
        get_in<1>() = other.template get_in<1>();
    }

    template <typename... Us>
    void copy_out(item_context<Rank, OutType(Us...)> &other)
    {
        get_out() = other.get_out();
    }

    item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }

    cl::sycl::item<Rank> get_item() const { return item_; }

private:
    std::tuple<T, Ts...> in_;
    OutType out_;
    cl::sycl::item<Rank> item_;
};

template <typename FirstContext, typename LastContext>
struct combined_context;

template <typename T, typename U, int Rank>
struct combined_context<item_context<Rank, T>, item_context<Rank, U>>
{
    using first_context_type = item_context<Rank, T>;
    using last_context_type = item_context<Rank, U>;

    using in_type = typename first_context_type::in_type;
    using out_type = typename last_context_type::out_type;

    using type = item_context<Rank, out_type(in_type)>;
};

template <typename T, typename U, int Rank>
struct combined_context<item_context<Rank, T(void)>, item_context<Rank, U>>
{
    using last_context_type = item_context<Rank, U>;
    using out_type = typename last_context_type::out_type;

    using type = item_context<Rank, out_type()>;
};

template <typename U, int Rank, typename OutType, typename T, typename... Ts>
struct combined_context<item_context<Rank, OutType(T, Ts...)>, item_context<Rank, U>>
{
    using last_context_type = item_context<Rank, U>;

    using type = item_context<Rank, typename last_context_type::out_type(T, Ts...)>;
};

template <typename FirstContext, typename LastContext>
using combined_context_t = typename combined_context<FirstContext, LastContext>::type;

template <typename Signatur>
struct item_context_from_signature;

template <typename OutType,
          typename FirstIn,
          typename SecondIn,
          int Rank,
          typename Any1,
          typename Any2,
          typename Any3>
struct item_context_from_signature<
    item_context<Rank, OutType(Any1)>(item_context<Rank, Any2(FirstIn)>, item_context<Rank, Any3(SecondIn)>)>
{
    using type = item_context<Rank, OutType(FirstIn, SecondIn)>;
};

template <typename OutType,
          typename InType,
          int Rank,
          typename Any1,
          typename Any2>
struct item_context_from_signature<
    item_context<Rank, OutType(Any1)>(item_context<Rank, Any2(InType)>)>
{
    using type = item_context<Rank, OutType(InType)>;
};

template <typename OutType,
          typename InType,
          int Rank,
          typename Any1,
          typename Any2,
          typename Any3>
struct item_context_from_signature<
    item_context<Rank, OutType(Any1)>(item_context<Rank, Any2(InType, Any3)>)>
{
    using type = item_context<Rank, OutType(InType)>;
};

template <typename Signature>
using item_context_from_signature_t = typename item_context_from_signature<Signature>::type;

} // namespace detail

namespace traits
{

template <typename T>
struct is_item_context : std::bool_constant<false>
{
};

template <int Rank, typename ContextType>
struct is_item_context<detail::item_context<Rank, ContextType>> : std::bool_constant<true>
{
};

template <typename T>
inline constexpr bool is_item_context_v = is_item_context<T>::value;

} // namespace traits

} // namespace celerity::algorithm

#endif // !ITEM_CONTEXT_H
