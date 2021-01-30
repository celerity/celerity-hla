#ifndef GENERIC_LAMBDA_TRAITS
#define GENERIC_LAMBDA_TRAITS

// #include <utility>
// #include <cstddef>

// constexpr size_t max_arity = 10;

// struct variadic_t
// {
// };

// namespace detail
// {
//     // it is templated, to be able to create a
//     // "sequence" of arbitrary_t's of given size and
//     // hece, to 'simulate' an arbitrary function signature.
//     template <size_t>
//     struct arbitrary_t
//     {
//         // this type casts implicitly to anything,
//         // thus, it can represent an arbitrary type.
//         template <typename T>
//         operator T &&();

//         template <typename T>
//         operator T &();
//     };

//     template<class T>
//     struct is_arbitrary_t : std::bool_constant<false> {};

//     template<size_t I>
//     struct is_arbitrary_t<arbitrary_t<I>> : std::bool_constant<true> {};

//     template <typename F, size_t... Is,
//               typename U = decltype(std::declval<F>()(arbitrary_t<Is>{}...))>
//     constexpr auto test_signature(std::index_sequence<Is...>)
//     {
//         return std::integral_constant<size_t, sizeof...(Is)>{};
//     }

//     template <size_t I, typename F>
//     constexpr auto arity_impl(int) -> decltype(test_signature<F>(std::make_index_sequence<I>{}))
//     {
//         return {};
//     }

//     template <size_t I, typename F>
//     constexpr auto arity_impl(...)
//     {
//         // try the int overload which will only work,
//         // if F takes I-1 arguments. Otherwise this
//         // overload will be selected and we'll try it
//         // with one element less.
//         static_assert(I > 0, "unable to deduce arity");
//         return arity_impl<I - 1, F>(0);
//     }

//     template <typename F, size_t MaxArity = 10>
//     constexpr auto arity_impl()
//     {
//         // start checking function signatures with max_arity + 1 elements
//         constexpr auto tmp = arity_impl<MaxArity + 1, F>(0);
//         if constexpr (tmp == MaxArity + 1)
//         {
//             // if that works, F is considered variadic
//             return variadic_t{};
//         }
//         else
//         {
//             // if not, tmp will be the correct arity of F
//             return tmp;
//         }
//     }
// } // namespace detail

// template <typename F, size_t MaxArity = max_arity>
// constexpr auto arity(F &&f) { return ::detail::arity_impl<std::decay_t<F>, MaxArity>(); }

// template <typename F, size_t MaxArity = max_arity>
// constexpr auto arity_v = detail::arity_impl<std::decay_t<F>, MaxArity>();

// template <typename F, size_t MaxArity = max_arity>
// constexpr bool is_variadic_v = std::is_same_v<std::decay_t<decltype(arity_v<F, MaxArity>)>, variadic_t>;

#endif // GENERIC_LAMBDA_TRAITS