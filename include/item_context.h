#ifndef ITEM_CONTEXT_H
#define ITEM_CONTEXT_H

#include "celerity_helper.h"

namespace celerity::hla
{

    namespace detail
    {

        template <int Rank, typename T>
        class item_shared_data
        {
        public:
            item_shared_data(const item_shared_data &rhs) = delete;
            item_shared_data(T &data, const cl::sycl::item<Rank> &item)
                : data_(data), item_(item) {}

            T &get() const { return data_; }

            operator cl::sycl::item<Rank>() const
            {
                return item_;
            }

            operator cl::sycl::id<Rank>() const
            {
                return item_.get_id();
            }

        private:
            T &data_;
            const cl::sycl::item<Rank> &item_;
        };

        template <typename Context>
        Context make_copy_in(const auto &ctx)
        {
            return Context{ctx.get_item(), ctx.template get_in<0>().get()};
        }

        template <typename Context>
        Context make_copy_out(const auto &ctx)
        {
            return Context{ctx.get_item(), ctx.get_out_value()};
        }

        template <typename Context>
        Context make_copy_out(const auto &ctxA, const auto &ctxB)
        {
            return Context{ctxA.get_item(), ctxA.get_out_value(), ctxB.get_out_value()};
        }

        template <typename Context>
        Context make_copy_in_out(const auto &ctxIn, const auto &ctxOut)
        {
            return Context{ctxIn.get_item(), ctxIn.template get_in<0>().get(), ctxOut.get_out_value()};
        }

        template <typename Context>
        Context make_out_only(const auto &ctx)
        {
            return Context{ctx.get_item()};
        }

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

            item_context(const item_context &) = delete;
            item_context &operator=(const item_context &) = delete;
            item_context(item_context &&) = delete;
            item_context &operator=(item_context &&) = delete;

            explicit item_context(const item_type &item)
                : item_(item) {}

            template <typename... Us>
            void copy_out(const item_context<Rank, OutType(Us...)> &other)
            {
                out_ = other.get_out_value();
            }

            item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }
            item_shared_data<Rank, const OutType> get_out() const { return {out_, item_}; }

            OutType get_out_value() const { return out_; }

            cl::sycl::item<Rank> get_item() const { return item_; }

        private:
            const cl::sycl::item<Rank> item_;
            OutType out_;
        };

        template <typename Context, int Rank, typename OutType>
        Context make_copy_in(const item_context<Rank, OutType()> &ctx)
        {
            return Context{ctx.get_item()};
        }

        template <int Rank, typename InType>
        class item_context<Rank, void(InType)>
        {
        public:
            using item_type = cl::sycl::item<Rank>;
            using in_type = InType;
            using out_type = void;

            static constexpr auto rank = Rank;

            item_context(const item_context &) = delete;
            item_context &operator=(const item_context &) = delete;
            item_context(item_context &&) = delete;
            item_context &operator=(item_context &&) = delete;

            explicit item_context(const item_type &item)
                : item_(item), in_() {}

            item_context(const cl::sycl::item<Rank> &item, InType in, auto)
                : item_(item), in_(in) {}

            template <typename ItemContext>
            explicit item_context(ItemContext &ctx)
                : item_(ctx.get_item()), in_(ctx.get_out_value())
            {
                // static_assert(std::is_convertible_v<std::decay_t<decltype(get_in())>, decltype(ctx.get_out())>);
            }

            template <typename ItemContextA, typename ItemContextB>
            item_context(ItemContextA a, ItemContextB b)
                : item_context(a)
            {
            }

            template <typename... Us>
            void copy_out(item_context<Rank, void(Us...)> &other)
            {
            }

            template <int Index = 0>
            item_shared_data<Rank, const InType> get_in() const { return {in_, item_}; }

            cl::sycl::item<Rank> get_item() const { return item_; }

            InType get_in_value() const { return in_; }

        private:
            const cl::sycl::item<Rank> item_;
            const InType in_;
        };

        template <int Rank, typename OutType, typename InType>
        class item_context<Rank, OutType(InType)>
        {
        public:
            using item_type = cl::sycl::item<Rank>;
            using in_type = InType;
            using out_type = OutType;

            static constexpr auto rank = Rank;

            item_context(const item_context &) = delete;
            item_context &operator=(const item_context &) = delete;
            item_context(item_context &&) = delete;
            item_context &operator=(item_context &&) = delete;

            explicit item_context(const item_type &item)
                : item_(item), in_() {}

            explicit item_context(cl::sycl::item<Rank> item, const in_type &in)
                : item_(item), in_(in) {}

            explicit item_context(cl::sycl::item<Rank> item, const in_type &in, auto)
                : item_(item), in_(in) {}

            template <typename... Us>
            void copy_out(item_context<Rank, OutType(Us...)> &other)
            {
                out_ = other.get_out_value();
            }

            template <int Index = 0>
            item_shared_data<Rank, const InType> get_in() const { return {in_, item_}; }

            item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }

            OutType get_out_value() const { return out_; }

            cl::sycl::item<Rank> get_item() const { return item_; }

        private:
            const cl::sycl::item<Rank> item_;
            const InType in_;
            OutType out_;
        };

        template <int Rank, typename OutType, typename T, typename... Ts>
        class item_context<Rank, OutType(T, Ts...)>
        {
        public:
            using item_type = cl::sycl::item<Rank>;

            using in_type = std::tuple<T, Ts...>;
            using out_type = OutType;

            static constexpr auto rank = Rank;

            item_context(const item_context &) = delete;
            item_context &operator=(const item_context &) = delete;
            item_context(item_context &&) = delete;
            item_context &operator=(item_context &&) = delete;

            explicit item_context(const item_type &item)
                : item_(item), in_() {}

            template <typename ItemContext>
            explicit item_context(const item_type &item, const auto &ctx)
                : item_(item), in_({ctx.get_out_value(), {}})
            {
                static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<0>())>, decltype(ctx.get_out())>);
            }

            item_context(const cl::sycl::item<Rank> &item, auto in_0)
                : item_(item), in_(in_0, {})
            {
            }

            item_context(const cl::sycl::item<Rank> &item, auto in_0, auto in_1)
                : item_(item), in_(in_0, in_1)
            {
                // static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<0>())>, decltype(in_0)>);
                // static_assert(std::is_convertible_v<std::decay_t<decltype(get_in<1>())>, decltype(in_1)>);
            }

            template <size_t Index>
            auto get_in() const
            {
                static_assert(Index < std::tuple_size_v<decltype(in_)>);
                return item_shared_data{std::get<Index>(in_), item_};
            }

            template <typename... Us>
            void copy_out(item_context<Rank, OutType(Us...)> &other)
            {
                out_ = other.get_out_value();
            }

            item_shared_data<Rank, OutType> get_out() { return {out_, item_}; }

            OutType get_out_value() const { return out_; }

            cl::sycl::item<Rank> get_item() const { return item_; }

        private:
            const cl::sycl::item<Rank> item_;
            const std::tuple<const T, const Ts...> in_;
            OutType out_;
        };

        template <int Rank, typename OutType, typename T, typename... Ts>
        item_context<Rank, OutType(T, Ts...)> make_copy_in(const item_context<Rank, OutType(T, Ts...)> &ctx)
        {
            return {ctx.get_item(), ctx.template get_in<0>().get(), ctx.template get_in<1>().get()};
        }

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

} // namespace celerity::hla

#endif // !ITEM_CONTEXT_H
