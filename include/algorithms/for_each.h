#ifndef FOR_EACH_H
#define FOR_EACH_H

#include "../iterator.h"
#include "../task.h"
#include "../accessor_proxy.h"
#include "../policy.h"
#include "../sequencing.h"
#include "../require.h"

namespace celerity::hla
{

    namespace detail
    {
        template <typename ExecutionPolicy, typename F, typename T, int Rank, template <typename, int> typename InIterator,
                  require<traits::get_accessor_type<F, 0>() == access_type::item> = yes>
        auto for_each_impl(InIterator<T, Rank> beg, InIterator<T, Rank> end, const F &f)
        {
            using namespace traits;
            using namespace cl::sycl::access;

            using policy_type = strip_queue_t<ExecutionPolicy>;

            if constexpr (!policy_traits<std::decay_t<ExecutionPolicy>>::is_distributed)
            {
                using accessor_type = hla::all<T, Rank>;

                return [=](celerity::handler &cgh) {
                    auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, accessor_type>(cgh, beg, end);

                    return [=]() {
                        for_each_index(beg, end, distance(beg, end), *beg, [f, in_acc](const cl::sycl::item<Rank> &item) {
                            f(item, in_acc[item][item]);
                        });
                    };
                };
            }
            else
            {
                using accessor_type = accessor_type_t<F, 0, T>;

                return [=](celerity::handler &cgh) {
                    auto in_acc = get_access<policy_type, cl::sycl::access::mode::read, accessor_type>(cgh, beg, end);

                    return [=]() {
                        for_each_index(beg, end, distance(beg, end), *beg, [f, in_acc](const cl::sycl::item<Rank> &item) {
                            f(item, in_acc[item][item]);
                        });
                    };
                };
            }
        }

        template <typename ExecutionPolicy, typename T, int Rank, typename F>
        auto for_each(buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
        {
            return task<ExecutionPolicy>(for_each_impl<ExecutionPolicy>(beg, end, f));
        }

    } // namespace detail

    template <typename ExecutionPolicy, typename T, int Rank, typename F,
              require<traits::get_accessor_type<F, 0>() == detail::access_type::item> = yes>
    auto for_each(ExecutionPolicy p, buffer_iterator<T, Rank> beg, buffer_iterator<T, Rank> end, const F &f)
    {
        return std::invoke(detail::for_each<ExecutionPolicy>(beg, end, f), p.q);
    }

    template <typename ExecutionPolicy, typename T, int Rank, typename F,
              require<traits::get_accessor_type<F, 0>() == detail::access_type::item> = yes>
    auto for_each(ExecutionPolicy p, buffer<T, Rank> in, const F &f)
    {
        return for_each(p, begin(in), end(in), f);
    }

} // namespace celerity::hla

#endif // FOR_EACH_H