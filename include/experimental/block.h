#ifndef CELERITY_HLA_BLOCK_H
#define CELERITY_HLA_BLOCK_H

#include "../sycl.h"
#include "../celerity_helper.h"
#include "../celerity_accessor_traits.h"

#include "function_traits.h"

namespace algo = celerity::hla;

namespace celerity::hla::experimental
{
    template <typename T, typename... Args>
    concept Callable = std::is_invocable_v<T, Args...>;

    template <typename T, size_t Rank>
    class block_probe
    {
    public:
        using value_type = T;
        static constexpr auto rank = Rank;

        block_probe() = default;

        void configure(cl::sycl::range<rank> size)
        {
            size_ = size;
            throw *this;
        }

        auto size() const -> cl::sycl::range<rank> { return size_; }
        auto item() const -> cl::sycl::item<rank> { return cl::sycl::detail::make_item<rank>({}, {}); }
        auto operator*() const -> value_type { return this->operator[]({}); }
        auto operator[](cl::sycl::rel_id<rank> rel_id) const -> value_type { return {}; }
        auto get(cl::sycl::id<rank> abs_id) const -> value_type { return {}; }
        auto is_on_boundary() const -> bool { return false; }
        auto is_on_boundary(cl::sycl::range<rank> range) const -> bool { return false; }

        template <Callable F>
        auto discern(F on_bounds_functor, Callable auto in_bounds_functor) const -> std::invoke_result_t<F> { return {}; }

        template <typename F, Callable G,
                  std::enable_if_t<std::is_same_v<F, std::invoke_result_t<G>>, int> = 0>
        auto discern(F on_bounds_value, G in_bounds_functor) const -> F { return {}; }

        template <Callable F, typename G,
                  std::enable_if_t<std::is_same_v<F, G, std::invoke_result_t<F>>, int> = 0>
        auto discern(F on_bounds_functor, G in_bounds_value) const -> G { return {}; }

    private:
        cl::sycl::range<rank> size_;
    };

    template <class Acc>
    class block
    {
    public:
        using accessor_type = Acc;
        using value_type = algo::traits::accessor_value_type_t<Acc>;
        static constexpr auto rank = algo::traits::accessor_rank_v<Acc>;

        template <typename AccessorType>
        block(const block_probe<value_type, rank> &probe, AccessorType acc, cl::sycl::item<rank> item)
            : item_(item), size_(probe.size()), acc_(acc)
        {
        }

        void configure(cl::sycl::range<rank>) {}
        auto size() const -> cl::sycl::range<rank> { return size_; }

        auto item() const { return item_; }

        value_type operator*() const
        {
            return this->operator[]({});
        }

        value_type operator[](cl::sycl::rel_id<rank> rel_id) const
        {
            auto id = item_.get_id();

            for (auto i = 0u; i < rank; ++i)
            {
                id[i] = static_cast<size_t>(static_cast<long>(id[i]) + rel_id[i]);
            }

            return acc_[id];
        }

        block(const block<Acc> &) = delete;
        block(block<Acc> &&) = delete;
        block<Acc> &operator=(const block<Acc> &) = delete;
        block<Acc> &operator=(block<Acc> &&) = delete;

        template <typename V>
        block<Acc> &operator=(const V &) = delete;

        value_type get(cl::sycl::id<rank> abs_id) const
        {
            return acc_[abs_id];
        }

        bool is_on_boundary() const
        {
            return dispatch_is_on_boundary(item_.get_range(), std::make_index_sequence<rank>());
        }

        bool is_on_boundary(cl::sycl::range<rank> range) const
        {
            return dispatch_is_on_boundary(range, std::make_index_sequence<rank>());
        }

        auto discern(Callable auto on_bounds_functor, Callable auto in_bounds_functor) const
        {
            return is_on_boundary()
                       ? std::invoke(on_bounds_functor)
                       : std::invoke(in_bounds_functor);
        }

        template <typename F, Callable G,
                  std::enable_if_t<std::is_same_v<F, std::invoke_result_t<G>>, int> = 0>
        auto discern(F on_bounds_value, G in_bounds_functor) const
        {
            return is_on_boundary()
                       ? on_bounds_value
                       : std::invoke(in_bounds_functor);
        }

        template <Callable F, typename G,
                  std::enable_if_t<std::is_same_v<F, G, std::invoke_result_t<F>>, int> = 0>
        auto discern(F on_bounds_functor, G in_bounds_value) const
        {
            return is_on_boundary()
                       ? std::invoke(on_bounds_functor)
                       : in_bounds_value;
        }

    private:
        const cl::sycl::item<rank> item_;
        const cl::sycl::range<rank> size_;
        accessor_type acc_;

        template <size_t... Is>
        bool dispatch_is_on_boundary(cl::sycl::range<rank> range, std::index_sequence<Is...>) const
        {
            const auto id = item_.get_id();

            return ((id[Is] < (size_[Is] / 2)) || ...) ||
                   ((static_cast<int>(id[Is]) > static_cast<int>(range[Is]) - static_cast<int>(size_[Is] / 2) - 1) || ...);
        }
    };

    template <typename T>
    struct is_block : std::bool_constant<false>
    {
    };

    template <typename Acc>
    struct is_block<block<Acc>> : std::bool_constant<true>
    {
    };

    template <typename T, size_t Rank>
    struct is_block<block_probe<T, Rank>> : std::bool_constant<true>
    {
    };

    template <typename T, size_t N, size_t... Is>
    inline constexpr bool are_equal(const std::array<T, N> &a, const std::array<T, N> &b, std::index_sequence<Is...>)
    {
        return ((a[Is] == b[Is]) && ...);
    }

    template <typename T>
    concept Block = is_block<T>::value || InactiveProbe<T>;

    template <typename T>
    concept StrictBlock = is_block<T>::value;

} // namespace celerity::hla::experimental

#endif // CELERITY_HLA_BLOCK_H