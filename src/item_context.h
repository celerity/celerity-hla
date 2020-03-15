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

private:
    T &data_;
    cl::sycl::item<Rank> item_;
};

template <int Rank, typename ContextType>
class item_context
{
public:
    using item_type = cl::sycl::item<Rank>;

    explicit item_context(cl::sycl::item<Rank> item)
        : item_(item) {}

    item_shared_data<Rank, ContextType> operator[](int idx)
    {
        if (switch_)
            return {shared_data_[1 - idx], item_};

        return {shared_data_[idx], item_};
    }

    cl::sycl::item<Rank> get_item() const { return item_; }

    void switch_data() { switch_ = !switch_; }

private:
    cl::sycl::item<Rank> item_;
    std::array<ContextType, 2> shared_data_;
    bool switch_ = false;
};

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
