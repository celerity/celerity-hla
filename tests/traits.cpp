#pragma clang diagnostic warning "-Wall"

#include "../include/accessor_proxy.h"
#include "../include/iterator.h"
#include "../include/kernel_traits.h"
#include "../include/sequence.h"
#include "../include/packaged_task_traits.h"
#include "../include/packaged_tasks/packaged_generate.h"
#include "../include/linkage_traits.h"

#include <vector>
#include <stdlib.h>

void static_assert_accessor_types()
{
    using namespace celerity::algorithm;
    using namespace celerity::algorithm::traits;
    using namespace celerity::algorithm::detail;

    auto access_one = [](int, int) {};
    using access_one_t = decltype(access_one);

    static_assert(get_accessor_type<access_one_t, 0>() == access_type::one_to_one);
    static_assert(get_accessor_type<access_one_t, 1>() == access_type::one_to_one);

    auto access_slice = [](slice<int, 0>, slice<int, 1>, slice<int, 2>) {};
    using access_slice_t = decltype(access_slice);

    static_assert(get_accessor_type<access_slice_t, 0>() == access_type::slice);
    static_assert(get_accessor_type<access_slice_t, 1>() == access_type::slice);
    static_assert(get_accessor_type<access_slice_t, 2>() == access_type::slice);

    using chunk_t = chunk<int, 1>;
    auto access_chunk = [](chunk_t, chunk_t, chunk_t) {};
    using access_chunk_t = decltype(access_chunk);

    static_assert(get_accessor_type<access_chunk_t, 0>() == access_type::chunk);
    static_assert(get_accessor_type<access_chunk_t, 1>() == access_type::chunk);
    static_assert(get_accessor_type<access_chunk_t, 2>() == access_type::chunk);

    using all_t = celerity::algorithm::all<int, 1>;
    auto access_all = [](all_t, all_t, all_t) {};
    using access_all_t = decltype(access_all);

    static_assert(get_accessor_type<access_all_t, 0>() == access_type::all);
    static_assert(get_accessor_type<access_all_t, 1>() == access_type::all);
    static_assert(get_accessor_type<access_all_t, 2>() == access_type::all);
}

void static_assert_iterator_traits()
{
    using namespace std;
    using namespace celerity;
    using namespace algorithm;

    const vector<float> v;

    static_assert(!traits::is_contiguous_iterator<decltype(begin(v))>());
    static_assert(traits::is_contiguous_iterator<decltype(v.data())>());
}

void static_assert_call_operator_detection()
{
    using namespace std;
    using namespace celerity;
    using namespace algorithm;

    auto foo = []() {};
    auto goo = [](handler &) {};
    auto hoo = [](auto) {};

    using foo_t = decltype(foo);
    using goo_t = decltype(goo);
    using hoo_t = decltype(hoo);

    static_assert(!algorithm::traits::has_call_operator_v<int>, "no call operator");
    static_assert(algorithm::traits::has_call_operator_v<goo_t>, "call operator");
    static_assert(algorithm::traits::has_call_operator_v<foo_t>, "call operator");
    static_assert(!algorithm::traits::has_call_operator_v<hoo_t>, "can not detect call operator templates");
}

void static_assert_kernel_traits()
{
    using namespace celerity::algorithm;
    using namespace celerity::algorithm::traits;
    using namespace celerity::algorithm::detail;

    auto one_d = [](item_context<1, int> &) {};
    auto two_d = [](item_context<2, int> &) {};
    auto three_d = [](item_context<3, int> &) {};
    //auto generic = [](auto) {};

    static_assert(is_kernel_v<decltype(one_d)>);
    static_assert(is_kernel_v<decltype(two_d)>);
    static_assert(is_kernel_v<decltype(three_d)>);
    //static_assert(!is_kernel_v<decltype(generic)>);

    auto compute_task = [one_d](celerity::handler &) { return one_d; };
    static_assert(is_compute_task_v<decltype(compute_task)>);
    static_assert(is_master_task_v<decltype(compute_task)>);

    auto master_task = [](celerity::handler &) { return []() {}; };
    static_assert(is_master_task_v<decltype(master_task)>);
    static_assert(!is_compute_task_v<decltype(master_task)>);

    sequence seq{[](celerity::distr_queue &) {}};
    static_assert(!is_packaged_task_v<decltype(seq)>);
    static_assert(!is_placeholder_task_v<decltype(seq), void>);

    using iterator_t = iterator<1>;
    auto placeholder = [](iterator_t, iterator_t) {};
    static_assert(is_placeholder_task_v<decltype(placeholder), iterator_t>);
    static_assert(!is_placeholder_task_v<decltype(placeholder), iterator<2>>);
}

#include "../include/experimental/accessor_proxies.h"
#include "../include/experimental/accessor_traits.h"
#include "../include/experimental/probing.h"
#include "../include/experimental/accessor_iterator.h"
#include "../include/policy.h"
#include "../include/experimental/kernel_traits.h"

void static_assert_kernel_probing()
{
    using namespace celerity;
    using namespace celerity::algorithm;
    using namespace celerity::hla::experimental;

    using value_t = int;
    using accessor_t = celerity::algorithm::detail::device_accessor<value_t, 1,
                                                                    cl::sycl::access::mode::read_write,
                                                                    cl::sycl::access::target::global_buffer>;

    {
        const auto f = [](auto x) { return x; };
        using f_t = decltype(f);

        static_assert(std::is_invocable_v<f_t, value_t>);
        static_assert(std::is_invocable_v<f_t, hla::experimental::slice<accessor_t>>);
    }

    // {
    //     const auto f = [](Slice<0> auto x) { return x; };
    //     using f_t = decltype(f);

    //     static_assert(!std::is_invocable_v<f_t, t_slice<int, 0>>);
    //     static_assert(std::is_invocable_v<f_t, hla::experimental::slice<accessor_t>>);
    // }

    // {
    //     const auto f = [](Slice<1> auto x) { return x; };
    //     using f_t = decltype(f);

    //     static_assert(!std::is_invocable_v<f_t, t_slice<int, 0>>);
    //     static_assert(!std::is_invocable_v<f_t, hla::experimental::slice<accessor_t, 0>>);
    // }

    static_assert(hla::experimental::traits::is_slice_v<hla::experimental::slice<accessor_t>>);
    static_assert(!hla::experimental::traits::is_block_v<hla::experimental::slice<accessor_t>>);
    static_assert(hla::experimental::traits::is_block_v<hla::experimental::block<accessor_t>>);

    {
        auto f = [](AnySlice auto x) {
            x.configure(0);
            return x[0];
        };

        using f_t = decltype(f);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 1, 0, slice_probe<int>, kernel_input<int, 1>>);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 1, 0, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::slice);

        celerity::buffer<int, 1> b{{10}};

        auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>>(f, begin(b), end(b));

        using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

        static_assert(std::is_same_v<proxy_type, hla::experimental::slice<accessor_t>>);
    }

    {
        auto f = [](AnyBlock auto x) {
            x.configure({0});
            return x[{0}];
        };

        using f_t = decltype(f);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 1, 0, block_probe<int, 1>, kernel_input<int, 1>>);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 1, 0, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::chunk);

        celerity::buffer<int, 1> b{{10}};

        auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>>(f, begin(b), end(b));

        using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

        static_assert(std::is_same_v<proxy_type, hla::experimental::block<accessor_t>>);
    }

    {
        auto f = [](AnyBlock auto x, AnySlice auto y) {
            x.configure({0});
            y.configure(0);

            return x[{0}] + y[0];
        };

        using f_t = decltype(f);

        static_assert(std::is_base_of_v<celerity::hla::experimental::inactive_probe_t, celerity::hla::experimental::concrete_inactive_probe<int>>);
        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 0, block_probe<int, 1>, kernel_input<int, 1>, kernel_input<int, 1>>);
        static_assert(!celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 0, slice_probe<int>, kernel_input<int, 1>, kernel_input<int, 1>>);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 1, slice_probe<int>, kernel_input<int, 1>, kernel_input<int, 1>>);
        static_assert(!celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 1, block_probe<int, 1>, kernel_input<int, 1>, kernel_input<int, 1>>);

        static_assert(celerity::hla::experimental::get_access_concept<f_t, 2, 0, kernel_input<int, 1>, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::chunk);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 2, 1, kernel_input<int, 1>, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::slice);

        celerity::buffer<int, 1> b{{10}};

        {
            auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>, kernel_input<int, 1>>(f, begin(b), end(b));

            using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

            static_assert(std::is_same_v<proxy_type, hla::experimental::block<accessor_t>>);
        }

        {
            auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<1, kernel_input<int, 1>, kernel_input<int, 1>>(f, begin(b), end(b));

            using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

            static_assert(std::is_same_v<proxy_type, hla::experimental::slice<accessor_t>>);
        }
    }

    {
        auto f = [](AnySlice auto x) {
            x.configure(0);

            const auto b = begin(x);
            const auto e = end(x);

            std::for_each(b, e, [](auto) {});

            return x[0];
        };

        using f_t = decltype(f);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 1, 0, slice_probe<int>, kernel_input<int, 1>>);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 1, 0, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::slice);

        celerity::buffer<int, 1> b{{10}};

        auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>>(f, begin(b), end(b));

        using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

        static_assert(std::is_same_v<proxy_type, hla::experimental::slice<accessor_t>>);
    }

    {
        auto f = [](AnyBlock auto x) {
            x.configure({0});

            const auto b = begin(x);
            const auto e = end(x);

            std::for_each(b, e, [](auto) {});

            return x[{0}];
        };

        using f_t = decltype(f);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 1, 0, block_probe<int, 1>, kernel_input<int, 1>>);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 1, 0, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::chunk);

        celerity::buffer<int, 1> b{{10}};

        auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>>(f, begin(b), end(b));

        using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

        static_assert(std::is_same_v<proxy_type, hla::experimental::block<accessor_t>>);
    }

    {
        auto f = [](AnyBlock auto x, AnySlice auto y) {
            x.configure({0});
            y.configure(0);

            std::for_each(begin(x), end(x), [](auto) {});
            std::for_each(begin(y), end(y), [](auto) {});

            return x[{0}] + y[0];
        };

        using f_t = decltype(f);

        static_assert(std::is_base_of_v<celerity::hla::experimental::inactive_probe_t, celerity::hla::experimental::concrete_inactive_probe<int>>);
        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 0, block_probe<int, 1>, kernel_input<int, 1>, kernel_input<int, 1>>);
        static_assert(!celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 0, slice_probe<int>, kernel_input<int, 1>, kernel_input<int, 1>>);

        static_assert(celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 1, slice_probe<int>, kernel_input<int, 1>, kernel_input<int, 1>>);
        static_assert(!celerity::hla::experimental::is_invocable_using_probes_v<f_t, 2, 1, block_probe<int, 1>, kernel_input<int, 1>, kernel_input<int, 1>>);

        static_assert(celerity::hla::experimental::get_access_concept<f_t, 2, 0, kernel_input<int, 1>, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::chunk);
        static_assert(celerity::hla::experimental::get_access_concept<f_t, 2, 1, kernel_input<int, 1>, kernel_input<int, 1>>() == celerity::algorithm::detail::access_type::slice);

        celerity::buffer<int, 1> b{{10}};

        {
            auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<0, kernel_input<int, 1>, kernel_input<int, 1>>(f, begin(b), end(b));

             using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

            static_assert(std::is_same_v<proxy_type, hla::experimental::block<accessor_t>>);
        }

        {
            auto [factory, _] = celerity::hla::experimental::create_proxy_factory_and_range_mapper<1, kernel_input<int, 1>, kernel_input<int, 1>>(f, begin(b), end(b));

            using proxy_type = std::invoke_result_t<decltype(factory), accessor_t, cl::sycl::item<1>>;

            static_assert(std::is_same_v<proxy_type, hla::experimental::slice<accessor_t>>);
        }

        using execution_policy_type = celerity::algorithm::detail::distributed_execution_policy;

        static_assert(!std::is_void_v<decltype(get_access<execution_policy_type, cl::sycl::access::mode::read, 0, decltype(begin(b)), decltype(begin(b))>(std::declval<celerity::handler &>(), begin(b), end(b), f))>);
        static_assert(!std::is_void_v<decltype(get_access<execution_policy_type, cl::sycl::access::mode::read, 1, decltype(begin(b)), decltype(begin(b))>(std::declval<celerity::handler &>(), begin(b), end(b), f))>);
    }
}

int main(int, char *[])
{
    static_assert_accessor_types();
    static_assert_iterator_traits();
    static_assert_call_operator_detection();
    static_assert_kernel_traits();

    return EXIT_SUCCESS;
}