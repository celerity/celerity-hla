#ifndef VARIANT_ITEM
#define VARIANT_ITEM

#include "sycl.h"
#include "platform.h"

#include <variant>

template <int... Ranks>
class variant_item
{
public:
    using variant_type = std::variant<
        cl::sycl::item<Ranks>...>;

    static_assert(((Ranks > 0 && Ranks < 4) || ...),
                  "invalid ranks. Must be either 1, 2 or 3");

    template <int Rank>
    explicit variant_item(cl::sycl::item<Rank> item)
        : rank_(Rank), var_(item)
    {
        static_assert(((Ranks == Rank) || ...), "invalid rank");
    }

    template <typename F>
    auto apply(F f) const
    {
        switch (rank_)
        {
        case 1:
            if constexpr (((Ranks == 1) || ...))
                return f(*std::get_if<cl::sycl::item<1>>(&var_));
            break;
        case 2:
            if constexpr (((Ranks == 2) || ...))
                return f(*std::get_if<cl::sycl::item<2>>(&var_));
            break;
        case 3:
            if constexpr (((Ranks == 3) || ...))
                return f(*std::get_if<cl::sycl::item<3>>(&var_));
            break;
        default:
            celerity::algorithm::detail::on_error("variant_item", "invalid rank");
        }

        return return_type_t<F>{};
    }

private:
    const int rank_;
    const variant_type var_;

    template <typename F, int Head, int... Tail>
    struct return_type
    {
        using type = std::invoke_result_t<F, cl::sycl::item<Head>>;
    };

    template <typename F>
    using return_type_t = typename return_type<F, Ranks...>::type;
};

#endif