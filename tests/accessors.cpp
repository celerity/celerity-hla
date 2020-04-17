#define CATCH_CONFIG_MAIN

#pragma clang diagnostic warning "-Wall"

#include <numeric>

#include "utils.h"
#include "../include/actions.h"

using namespace celerity;
using namespace celerity::algorithm;
using namespace celerity::algorithm::aliases;

SCENARIO("accessing a slice", "[accessors::slice]")
{
    distr_queue q;

    GIVEN("Two two-dimensional buffers of 64x64 one representing the identity matrix and the other identity * 2")
    {
        constexpr auto rank = 16;

        std::vector<int> identity(rank * rank);
        for (size_t i = 0; i < rank; ++i)
        {
            for (size_t j = 0; j < rank; ++j)
            {
                identity[i * rank + j] = (i == j) * 3;
            }
        }

        buffer<int, 2> buf_a(identity.data(), {rank, rank});

        for (size_t i = 0; i < rank; ++i)
        {
            for (size_t j = 0; j < rank; ++j)
            {
                identity[i * rank + j] = (i == j) * 5;
            }
        }

        buffer<int, 2> buf_b(identity.data(), {rank, rank});

        WHEN("multiplying")
        {
            buffer<int, 2> buf_c(buf_a.get_range());

            transform<class matrix_mul>(q, buf_a, buf_b, buf_c,
                                        [](const slice_i<1> &a, const slice_i<0> &b) {
                                            auto sum = 0;

                                            for (auto k = 0; k < rank; ++k)
                                            {
                                                const auto a_ik = a[k];
                                                const auto b_kj = b[k];

                                                sum += a_ik * b_kj;
                                            }

                                            return sum;
                                        });

            THEN("the result is the identity matrix times 8")
            {
                const auto r = copy_to_host(q, buf_c);

                for (size_t i = 0; i < rank; ++i)
                {
                    for (size_t j = 0; j < rank; ++j)
                    {
                        const float correct_value = (i == j) * 15;

                        REQUIRE(r[j * rank + i] == correct_value);
                    }
                }
            }
        }
    }
}

SCENARIO("accessing a chunk", "[accessors::chunk]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of a hundred 1s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf(cl::sycl::range<1>{size});
        fill<class fill_x>(q, buf, 1);

        WHEN("adding chunks of 3")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class add_chunk>(q, buf, buf_out, [r = buf.get_range()](const chunk_i<3> &x) {
                if (x.is_on_boundary(r))
                    return 2;

                return x[{-1}] + *x + x[{1}];
            });

            THEN("the first and last element are 2, the others are 3")
            {
                const auto r = copy_to_host(q, buf_out);

                on_master([&]() {
                    REQUIRE(r.front() == 2);
                    REQUIRE(r.back() == 2);
                    REQUIRE(elements_equal_to<3>(next(begin(r)), prev(end(r))));
                });
            }
        }
    }

    GIVEN("Two two-dimensional buffer of a hundred 1s and a hundred 2s")
    {
        constexpr auto size = 100;

        buffer<int, 1> buf_a(cl::sycl::range<1>{size});
        buffer<int, 1> buf_b(cl::sycl::range<1>{size});

        fill<class fill_a>(q, buf_a, 2);
        fill<class fill_b>(q, buf_b, 5);

        REQUIRE(elements_equal_to<2>(copy_to_host(q, buf_a)));
        REQUIRE(elements_equal_to<5>(copy_to_host(q, buf_b)));

        WHEN("summing two chunks of 3x1")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            transform<class sum_chunks_3x3>(q, buf_a, buf_b, buf_out, [r = buf_a.get_range()](const chunk_i<3> &a, const chunk_i<3> &b) {
                if (a.is_on_boundary(r))
                    return 14;

                auto sum = 0;

                for (int i = -1; i < 2; ++i)
                {
                    sum += a[{i}] + b[{i}];
                }

                return sum;
            });

            THEN("the first and last element are 6, the others are 9")
            {
                const auto r = copy_to_host(q, buf_out);

                on_master([&]() {
                    REQUIRE(r.front() == 14);
                    REQUIRE(r.back() == 14);
                    REQUIRE(elements_equal_to<21>(next(begin(r)), prev(end(r))));
                });
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 1s")
    {
        constexpr auto rank = 10;

        buffer<int, 2> buf({rank, rank});
        fill<class fill_10x10>(q, buf, 1);

        WHEN("")

        WHEN("summing chunks of 3x3")
        {
            constexpr auto chunk_size = 3;

            buffer<int, 2> buf_out(buf.get_range());

            transform<class sum_chunk_3x3>(q, buf, buf_out, [r = buf.get_range()](const chunk_i<chunk_size, chunk_size> &c) {
                if (c.is_on_boundary(r))
                    return 0;

                auto sum = 0;

                for (auto y = -(chunk_size / 2); y <= chunk_size / 2; ++y)
                {
                    for (auto x = -(chunk_size / 2); x <= chunk_size / 2; ++x)
                    {
                        sum += c[{y, x}];
                    }
                }

                return sum;
            });

            THEN("border elements are 0, the others are 9")
            {
                const auto r = copy_to_host(q, buf_out);

                on_master([&]() {
                    for (size_t y = 0; y < rank; ++y)
                    {
                        for (size_t x = 0; x < rank; ++x)
                        {
                            const auto correct_value = (x == 0 || x == rank - 1 ||
                                                        y == 0 || y == rank - 1)
                                                           ? 0
                                                           : 9;

                            REQUIRE(r[y * rank + x] == correct_value);
                        }
                    }
                });
            }
        }
    }
}

SCENARIO("accessing the entire buffer", "[accessors::all]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer of one 2 and ninety-nine 1s")
    {
        constexpr size_t size = 100;

        std::vector<int> src(size, 1);
        src.front() = 2;

        buffer<int, 1> buf(src.data(), cl::sycl::range<1>{size});

        WHEN("assigning all elements the value of the first")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class set_first>(q, buf, buf_out, [](const all_i &x) {
                return x[{0}];
            });

            THEN("each element is equal to 2")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<2>(r));
            }
        }

        WHEN("assigning the sum of all elements to all elements")
        {
            buffer<int, 1> buf_out(buf.get_range());

            transform<class set_sum>(q, buf, buf_out, [r = buf.get_range()](const all_i &x) {
                auto sum = 0;

                for (size_t i = 0; i < size; ++i)
                    sum += x[{i}];

                return sum;
            });

            THEN("each element is equal to 101")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<101>(r));
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 elements where the first is 2 and the others are 1")
    {
        constexpr size_t rank = 10;

        std::vector<int> src(rank * rank, 1);
        src.front() = 2;

        buffer<int, 2> buf(src.data(), cl::sycl::range<2>{rank, rank});

        WHEN("assigning all elements the value of the first")
        {
            buffer<int, 2> buf_out(buf.get_range());

            transform<class set_first_10x10>(q, buf, buf_out, [](const all_i2 &x) {
                return x[{0, 0}];
            });

            THEN("each element is equal to 2")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<2>(r));
            }
        }
    }

    GIVEN("A two-dimensional buffer of 10x10 1s")
    {
        constexpr size_t rank = 10;

        std::vector<int> src(rank * rank, 1);
        buffer<int, 2> buf(src.data(), cl::sycl::range<2>{rank, rank});

        WHEN("applying a 3x3 filter of 1s and a 2 at the center")
        {
            constexpr size_t filter_size = 3;

            std::vector<int> filter_src(filter_size * filter_size, 1);
            filter_src[filter_src.size() / 2] = 2;

            buffer<int, 2> buf_filter(filter_src.data(), {filter_size, filter_size});
            buffer<int, 2> buf_out(buf.get_range());

            using chunk_t = chunk_i<filter_size, filter_size>;

            transform<class apply_filter>(q, buf, buf_filter, buf_out, [r = buf.get_range(), fs = static_cast<int>(filter_size)](const chunk_t &in, const all_i2 &filter) {
                if (in.is_on_boundary(r))
                    return 0;

                auto sum = 0;

                for (auto y = -(fs / 2); y <= fs / 2; ++y)
                {
                    for (auto x = -(fs / 2); x <= fs / 2; ++x)
                    {
                        sum += filter[{static_cast<size_t>(fs / 2 + y), static_cast<size_t>(fs / 2 + x)}] * in[{y, x}];
                    }
                }

                return sum;
            });

            THEN("border elements are 0 and the others are 10")
            {
                const auto r = copy_to_host(q, buf_out);
                for (size_t y = 0; y < rank; ++y)
                {
                    for (size_t x = 0; x < rank; ++x)
                    {
                        const auto correct_value = (x == 0 || x == rank - 1 ||
                                                    y == 0 || y == rank - 1)
                                                       ? 0
                                                       : 10;

                        REQUIRE(r[y * rank + x] == correct_value);
                    }
                }
            }
        }
    }
}

SCENARIO("using any_accessor<T>", "[accessors::any_accessor]")
{
    distr_queue q;

    GIVEN("A one-dimensional buffer a hundred 1s")
    {
        constexpr auto size = 100;

        std::vector<int> src(size, 1);
        buffer<int, 1> buf(src.data(), cl::sycl::range<1>{size});

        WHEN("doubling every element")
        {
            buffer<int, 1> buf_out(buf.get_range());

            q.submit([=](handler &cgh) {
                auto in = buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
                auto out = buf_out.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());

                cgh.parallel_for<class doubling>(buf_out.get_range(), [=](cl::sycl::item<1> it) {
                    out[it] = in[it] * 2;
                });
            });

            THEN("each element is equal to 2")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<2>(r));
            }
        }

        WHEN("doubling every element using any_accessor<T>")
        {
            buffer<int, 1> buf_out(buf.get_range());

            const auto double_it = [](cl::sycl::item<1> it, const celerity::algorithm::detail::any_accessor<int> &in) {
                return in.get(it.get_id()) * 2;
            };

            q.submit([=](handler &cgh) {
                auto in = buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
                auto out = buf_out.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());

                cgh.parallel_for<class doubling_any>(buf_out.get_range(), [=](cl::sycl::item<1> it) {
                    auto any_in = celerity::algorithm::detail::any_accessor<int>(in);
                    out[it] = double_it(it, any_in);
                });
            });

            THEN("each element is equal to 2")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<2>(r));
            }
        }
    }

    GIVEN("Two one-dimensional buffers of hundred 1s and a hundred 5s")
    {
        constexpr auto size = 100;

        std::vector<int> src_a(size, 1);
        std::vector<int> src_b(size, 5);

        buffer<int, 1> buf_a(src_a.data(), cl::sycl::range<1>{size});
        buffer<int, 1> buf_b(src_b.data(), cl::sycl::range<1>{size});

        WHEN("adding")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](cl::sycl::item<1> i, auto a, auto b) {
                return a[i] + b[i];
            };

            q.submit([=](handler &cgh) {
                auto in_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
                auto in_b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());

                cgh.parallel_for<class adding_ab>(buf_out.get_range(), [=](cl::sycl::item<1> it) {
                    out[it] = add(it, in_a, in_b);
                });
            });

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("adding using any_accessor<T>")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](cl::sycl::item<1> i, const auto &a, auto b) {
                return a.get(i.get_id()) + b.get(i.get_id());
            };

            q.submit([=](handler &cgh) {
                auto in_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
                auto in_b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());

                cgh.parallel_for<class adding_ab_any>(buf_out.get_range(), [=](cl::sycl::item<1> it) {
                    out[it] = add(it,
                                  algorithm::detail::any_accessor<int>(in_a),
                                  algorithm::detail::any_accessor<int>(in_b));
                });
            });

            /* 
            // TODO: try different clang versions
            //       try computecpp 
            //
            // since this works, it seems likely there is something going wrong on the device or
            // while compiling for it

            const auto add = [](int i, auto a, auto b) {
                return a.get<1>({static_cast<size_t>(i)}) + b.get<1>({static_cast<size_t>(i)});
            };

            q.with_master_access([&](handler &cgh) {
                auto in_a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, buf_a.get_range());
                auto in_b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, buf_b.get_range());

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(cgh, buf_out.get_range());

                cgh.run([&]() {
                    auto any_in_a = celerity::detail::any_accessor<int>(in_a);
                    auto any_in_b = celerity::detail::any_accessor<int>(in_b);

                    for (auto i = 0; i < size; ++i)
                    {
                        out[i] = add(i, any_in_a, any_in_b);
                    }
                });
            });*/

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("add(chunk_i<1>, chunk_i<1>)")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](chunk_i<1> a, chunk_i<1> b) {
                // *b returns the same as *a
                return *a + *b;
            };

            using namespace celerity::algorithm::detail;

            const auto initialize = [=](handler &c) {
                auto in_a = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_a), end(buf_a));

                auto in_b = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_b), end(buf_b));

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(c, celerity::access::one_to_one<1>());

                return [=](item_context<1, int(int, int)> &it) {
                    out[it.get_out()] = add(in_a[it.get_in<0>()], in_b[it.get_in<1>()]);
                };
            };

            auto t = task<named_distributed_execution_and_queue_policy<class adding_ab_proxy_1>>(initialize);

            t(q, begin(buf_out), end(buf_out));

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("add(chunk_i<1>, int)")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](chunk_i<1> a, int b) {
                return *a + b;
            };

            using namespace celerity::algorithm::detail;

            const auto initialize = [=](handler &c) {
                auto in_a = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_a), end(buf_a));

                auto in_b = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_b), end(buf_b));

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(c, celerity::access::one_to_one<1>());

                return [=](item_context<1, int(int, int)> &it) {
                    out[it.get_out()] = add(in_a[it.get_in<0>()], *in_b[it.get_in<1>()]);
                };
            };

            auto t = task<named_distributed_execution_and_queue_policy<class adding_ab_proxy_2>>(initialize);

            t(q, begin(buf_out), end(buf_out));

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("add(int, chunk_i<1>)")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](int a, chunk_i<1> b) {
                return a + *b;
            };

            using namespace celerity::algorithm::detail;

            const auto initialize = [=](handler &c) {
                auto in_a = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_a), end(buf_a));

                auto in_b = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_b), end(buf_b));

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(c, celerity::access::one_to_one<1>());

                return [=](item_context<1, int(int, int)> &it) {
                    out[it.get_out()] = add(*in_a[it.get_in<0>()], in_b[it.get_in<1>()]);
                };
            };

            auto t = task<named_distributed_execution_and_queue_policy<class adding_ab_proxy_3>>(initialize);

            t(q, begin(buf_out), end(buf_out));

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("add(int, int)")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](int a, int b) {
                return a + b;
            };

            using namespace celerity::algorithm::detail;

            const auto initialize = [=](handler &c) {
                auto in_a = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<3>>(c, begin(buf_a), end(buf_a));

                auto in_b = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<3>>(c, begin(buf_b), end(buf_b));

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(c, celerity::access::one_to_one<1>());

                return [=](item_context<1, int(int, int)> &it) {
                    out[it.get_out()] = add(*in_a[it.get_in<0>()], *in_b[it.get_in<1>()]);
                };
            };

            auto t = task<named_distributed_execution_and_queue_policy<class adding_ab_proxy_4>>(initialize);

            t(q, begin(buf_out), end(buf_out));

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<6>(r));
            }
        }

        WHEN("add(const chunk_i<1>&, const chunk_i<1>&)")
        {
            buffer<int, 1> buf_out(buf_a.get_range());

            const auto add = [](const chunk_i<1> &a, const chunk_i<1> &b) {
                return *a + *b;
            };

            using namespace celerity::algorithm::detail;

            const auto initialize = [=](handler &c) {
                auto in_a = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_a), end(buf_a));

                auto in_b = get_access<distributed_execution_policy,
                                       cl::sycl::access::mode::read, chunk_i<1>>(c, begin(buf_b), end(buf_b));

                auto out = buf_out.get_access<cl::sycl::access::mode::write>(c, celerity::access::one_to_one<1>());

                return [=](item_context<1, int(int, int)> &it) {
                    out[it.get_out()] = add(in_a[it.get_in<0>()], in_b[it.get_in<1>()]);
                };
            };

            auto t = task<named_distributed_execution_and_queue_policy<class adding_ab_proxy_5>>(initialize);

            t(q, begin(buf_out), end(buf_out));

            THEN("each element is equal to 6")
            {
                const auto r = copy_to_host(q, buf_out);
                std::cout << r[0] << std::endl;
                REQUIRE(elements_equal_to<6>(r));
            }
        }
    }
}