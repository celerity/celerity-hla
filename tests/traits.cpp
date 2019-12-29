#include "../src/accessor_proxy.h"
#include "../src/iterator.h"

#include <vector>
#include <stdlib.h>

void static_assert_accessor_types()
{
    using namespace celerity::algorithm;
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

    using all_t = all<int, 1>;
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

    static_assert(!is_contiguous_iterator<decltype(begin(v))>());
    static_assert(is_contiguous_iterator<decltype(v.data())>());
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

    static_assert(!algorithm::detail::has_call_operator_v<int>, "no call operator");
    static_assert(algorithm::detail::has_call_operator_v<goo_t>, "call operator");
    static_assert(algorithm::detail::has_call_operator_v<foo_t>, "call operator");
    static_assert(!algorithm::detail::has_call_operator_v<hoo_t>, "can not detect call operator templates");
}

int main(int, char *[])
{
    static_assert_accessor_types();
    static_assert_iterator_traits();
    static_assert_call_operator_detection();

    return EXIT_SUCCESS;
}