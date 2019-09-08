#define MOCK_CELERITY
#define _ITERATOR_DEBUG_LEVEL 0

#include "../../src/sequence.h"
#include "../../src/actions.h"
#include "../../src/task_sequence.h"
#include "../../src/kernel_sequence.h"
#include "../../src/kernel_traits.h"
#include "../../src/static_iterator.h"
#include "../../src/algorithm.h"

#include <iostream>

using namespace std;

void sequence_static_assertions()
{
	using namespace celerity;
	using namespace algorithm;

	auto hello_world = []() { std::cout << "hello world" << endl; };
	auto zero = [](handler) {};

	using hello_world_t = decltype(hello_world);
	using zero_t = decltype(zero);


	static_assert(std::is_same<
		decltype(zero | task(zero)),
		sequence<task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(zero | zero),
		kernel_sequence<zero_t, zero_t>>::value,
		"actions not promoted to kernel_sequence");

	static_assert(std::is_same<
		decltype(zero | zero | zero),
		kernel_sequence<zero_t, zero_t, zero_t>>::value,
		"action not appended to kernel_sequence");

	/*static_assert(std::is_same<
		decltype(hello_world | zero),
		sequence<hello_world_t, task_t<distributed_execution_policy, zero_t>>>::value,
		"action not promoted to task_t");*/

	static_assert(std::is_same<
		decltype(zero | hello_world),
		sequence<task_t<distributed_execution_policy, zero_t>, hello_world_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(task(zero)),
		task_t<distributed_execution_policy, zero_t>>::value,
		"action not promoted to task_t");

	static_assert(std::is_same<
		decltype(fuse(zero | zero)),
		task_t<distributed_execution_policy, zero_t, zero_t>>::value,
		"actions to fused");

	static_assert(std::is_same<
		decltype(fuse(zero | zero | zero)),
		task_t<distributed_execution_policy, zero_t, zero_t, zero_t>>::value,
		"actions to fused");

	static_assert(std::is_same<
		decltype(zero | zero | task(zero)),
		sequence<task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>>>::value,
		"action sequence not promoted to task_t sequence");

  	static_assert(std::is_same<
		decltype(zero | task(zero) | zero),
		sequence<task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>>>::value,
		"action not promoted to task_t");

  	/*static_assert(std::is_same<
		decltype(hello_world | zero | task(zero) | zero),
		sequence<hello_world_t, task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>, task_t<distributed_execution_policy, zero_t>>>::value,
		"action not promoted to task_t");*/

	static_assert(!algorithm::detail::has_call_operator_v<int>, "no call operator");
	static_assert(algorithm::detail::has_call_operator_v<decltype(zero)>, "no call operator");
	static_assert(algorithm::detail::has_call_operator_v<decltype(hello_world)>, "no call operator");
	static_assert(algorithm::detail::get_accessor_type<decltype(zero), 0>() == access_type::one_to_one, "get_accessor_type");
	static_assert(algorithm::detail::get_accessor_type<algorithm::buffer_iterator<float, 1>, 0>() == access_type::invalid, "get_accessor_type");


	static_assert(!is_contiguous_iterator<decltype(begin(std::declval<vector<float>>()))>());
	static_assert(is_contiguous_iterator<decltype(std::declval<vector<float>>().data())>());
}



template<typename BufferType>
struct slice_of : celerity::algorithm::slice<typename BufferType::value_type, BufferType::rank>
{
};

#define slice(b) slice_of<decltype(b)>

void sequence_examples()
{
	// example 1: generic action sequence

	using namespace celerity;
	using namespace algorithm;
	using namespace actions;

	auto i = 0;
	auto seq = hello_world() | incr(i) | incr(i) | incr(i);
	std::invoke(seq);
	cout << i << endl << endl;

	// example 2: celerity task sequence

	auto zero = [](handler cgh)
	{
		/*
		auto dw_buf = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
			cgh.parallel_for<class zero>(buf.get_range(), [=](cl::sycl::item<2> item) { dw_buf[item] = 0.f; });
		*/

		cout << cgh.invocations << ": zero" << endl;
	};

	auto step = [](handler cgh)
	{
		/*
			auto rw_up = up.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
			auto r_u = u.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));

			const auto size = up.get_range();
			cgh.parallel_for<KernelName>(size, [=](cl::sycl::item<2> item) {
				// NOTE: We have to do some casting due to some weird ComputeCpp behavior - possibly a device compiler bug.
				// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-94 (psalz)
				const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
				const size_t my = item[0] > 0 ? static_cast<size_t>(static_cast<int>(item[0]) - 1) : item[0];
				const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
				const size_t mx = item[1] > 0 ? static_cast<size_t>(static_cast<int>(item[1]) - 1) : item[1];

				// NOTE: We have to copy delta here, again to avoid some ComputeCpp weirdness.
				cl::sycl::float2 delta2 = delta;
				const float lap = (dt / delta2.y()) * (dt / delta2.y()) * ((r_u[{py, item[1]}] - r_u[item]) - (r_u[item] - r_u[{my, item[1]}]))
								  + (dt / delta2.x()) * (dt / delta2.x()) * ((r_u[{item[0], px}] - r_u[item]) - (r_u[item] - r_u[{item[0], mx}]));
				rw_up[item] = Config::a * 2 * r_u[item] - Config::b * rw_up[item] + Config::c * lap;
			});
		*/

		cout << cgh.invocations << ": step" << endl;
	};

	distr_queue q{};

    zero | fuse(step | step | step) | step | submit_to(q);

	std::cout << endl;

	// algorithms

	using namespace algorithm;

	buffer<float, 1> b{ { 5 } };
	buffer<float, 1> c{ { 5 } };
	buffer<float, 1> b_out{ { 5 } };

	static_assert(std::is_base_of_v<slice<float, 1>, slice_of<decltype(b)>>);

	algorithm::fill(algorithm::distr<class fill>(q), begin(b), end(b), 1.f);
	algorithm::transform(algorithm::distr<class mul2>(q), next(begin(b)), end(b), begin(c), [](const float x) { return 2 * x; });
	
	assert(c.data()[0] == 1 && c.data()[1] == 2 && c.data()[2] == 2 && c.data()[3] == 2 && c.data()[4] == 2);
	
	algorithm::transform(algorithm::distr<class sum>(q), begin(b), end(b), begin(b_out),
		[&](slice<float, 0> x)
		{
			auto sum = *x;

			for (int i = 0; i < 5; ++i)
				sum += x[{i}];

			return sum;
		});

	algorithm::transform(master(q), begin(b), end(b), begin(b_out),
		[](float x)
		{
			return 2 * x;
		});

	/*algorithm::transform(master(q), begin(b), end(b), begin(b_out),
		[](chunk<float, 2> x)
		{
			return 2 * x[{0}];
		});*/

	algorithm::transform(algorithm::distr<class product>(q), begin(b), end(b), begin(c), begin(b_out),
		[](float x, float y)
		{
			return x * y;
		});
	
	algorithm::transform(algorithm::master(q), begin(b), end(b), begin(c), begin(b_out),
		[](float x, float y)
		{
			return x * y;
		});

	// algorithm action

	auto add_one = actions::transform(distr<class add_one>(q), begin(b), end(b), begin(b_out), [](float x) { return x + 1.0f; });

	zero | fuse(step | step | step) | step | add_one | submit_to(q);
}

void iterator_static_assertions()
{
	using namespace celerity::algorithm::fixed;

	static_assert(static_index<1>::rank == 1, "static_index rank");
	static_assert(static_index<1, 2, 3>::rank == 3, "static_index rank");

	static_assert(static_index<1>::components[0] == 1, "static_index::components");
	static_assert(static_index<1, 2>::components[0] == 1 &&
		static_index<1, 2>::components[1] == 2, "static_index::components");

	static_assert(is_same<static_index<1>, static_index<1>>::value, "static_index equality");
	static_assert(is_same<static_index<1, 3>, static_index<1, 3>>::value, "static_index equality");
	static_assert(!is_same<static_index<1, 3>, static_index<1, 2>>::value, "static_index inequality");

	static_assert(static_iterator<float, 1, 2, 3>::rank == 3, "static_iterator::rank");
	static_assert(is_same<static_iterator<float, 1, 2, 3>::value_type, float>::value, "static_iterator value_type");
	static_assert(is_same<static_iterator<float, 1, 2, 3>::index_type, static_index<1, 2, 3>>::value, "static_iterator::value_type");
	static_assert(is_same<static_iterator<float, 1, 2, 3>, static_iterator<float, 1, 2, 3>>::value, "static_iterator equality");
	static_assert(!is_same<static_iterator<float, 1, 2, 3>, static_iterator<float, 3, 3, 3>>::value, "static_iterator inequality");


	static_assert(static_view<1, static_iterator<float, 0>, static_iterator<float, 0>>::id == 1, "static_view::id");
	static_assert(static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>::rank == 2, "static_view::rank");

	static_assert(is_same<static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>,
		static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>>::value,
		"static_view::rank");

	static_assert(!is_same<static_view<1, static_iterator<float, 0, 0>, static_iterator<float, 1, 1>>,
		static_view<1, static_iterator<float, 0, 1>, static_iterator<float, 1, 1>>>::value,
		"static_view::rank");

	static_assert(is_same<decltype(begin(celerity::buffer<float, 1>{{1}})), static_iterator<float, 0 >> ::value, "begin");

	static_assert(is_same<decltype(end(celerity::buffer<float, 1>{{1}})), static_iterator<float, 0 >> ::value, "end");

	static_assert(is_same<decltype(make_view<1>(celerity::buffer<float, 1>{{1}})), static_view<1, static_iterator<float, 0>, static_iterator<float, 0>> > ::value, "make_view return type");

	using namespace celerity;
	using namespace algorithm;

	const auto src_view = fixed::make_view<1>(celerity::buffer<float, 1>{{1}});
	const auto dst_view = fixed::make_view<2>(celerity::buffer<float, 1>{{1}});
}

void sycl_helper_assertions()
{
	using namespace celerity;
	using namespace cl::sycl;

	/*static_assert(count(cl::sycl::range<1> { 2 }) == 2, "count");
	static_assert(count(cl::sycl::range<1> { 3 }) == 3, "count");
	static_assert(count(cl::sycl::range<2> { 1, 2 }) == 2, "count");
	static_assert(count(cl::sycl::range<2> { 2, 2 }) == 4, "count");
	static_assert(count(cl::sycl::range<3> { 2, 2, 2 }) == 8, "count");
	static_assert(count(cl::sycl::range<3> { 2, 3, 2 }) == 12, "count");

	static_assert(equals(next(cl::sycl::id<1> { 0 }, max_id(cl::sycl::range<1> { 2 })), cl::sycl::id<1> { 1 }), "next");
	static_assert(equals(next(cl::sycl::id<1> { 1 }, max_id(cl::sycl::range<1> { 2 })), cl::sycl::id<1> { 2 }), "next");
	static_assert(equals(next(cl::sycl::id<1> { 2 }, max_id(cl::sycl::range<1> { 2 })), cl::sycl::id<1> { 3 }), "next");
	static_assert(equals(next(cl::sycl::id<2> { 0, 0 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 0, 1 }), "next");
	static_assert(equals(next(cl::sycl::id<2> { 0, 0 }, max_id(cl::sycl::range<2> { 2, 1 })), cl::sycl::id<2> { 1, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<2> { 0, 1 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 1, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 1 }, max_id(cl::sycl::range<3> { 2, 2, 2 })), cl::sycl::id<3> { 0, 1, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 1, 1 }, max_id(cl::sycl::range<3> { 2, 2, 2 })), cl::sycl::id<3> { 1, 0, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, max_id(cl::sycl::range<3> { 2, 2, 2 }), 2), cl::sycl::id<3> { 0, 1, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, max_id(cl::sycl::range<3> { 2, 2, 2 }), 3), cl::sycl::id<3> { 0, 1, 1 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, max_id(cl::sycl::range<3> { 2, 2, 2 }), 4), cl::sycl::id<3> { 1, 0, 0 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, max_id(cl::sycl::range<3> { 2, 2, 2 }), 7), cl::sycl::id<3> { 1, 1, 1 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 7), cl::sycl::id<3> { 0, 2, 1 }), "next");
	static_assert(equals(next(cl::sycl::id<3> { 2, 2, 2 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 1), cl::sycl::id<3> { 3, 0, 0 }), "next");
	
	static_assert(equals(prev(cl::sycl::id<1> { 2 }, max_id(cl::sycl::range<1> { 2 })), cl::sycl::id<1> { 1 }), "prev");
	static_assert(equals(prev(cl::sycl::id<2> { 0, 1 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 0, 0 }), "prev");
	static_assert(equals(prev(cl::sycl::id<2> { 1, 0 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 0, 1 }), "prev");
	static_assert(equals(prev(cl::sycl::id<2> { 1, 1 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 1, 0 }), "prev");
	static_assert(equals(prev(cl::sycl::id<3> { 2, 2, 0 }, max_id(cl::sycl::range<3> { 3, 3, 3 })), cl::sycl::id<3> { 2, 1, 2 }), "prev");
	static_assert(equals(prev(cl::sycl::id<3> { 2, 0, 0 }, max_id(cl::sycl::range<3> { 3, 3, 3 })), cl::sycl::id<3> { 1, 2, 2 }), "prev");
	static_assert(equals(prev(cl::sycl::id<3> { 2, 2, 3 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 2), cl::sycl::id<3> { 2, 2, 1 }), "prev");
	static_assert(equals(prev(cl::sycl::id<3> { 2, 2, 2 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 3), cl::sycl::id<3> {2, 1, 2 }), "prev");

	static_assert(equals(max_id(cl::sycl::range<3> { 2, 2, 2 }), cl::sycl::id<3> { 1, 1, 1 }), "max_id");
	static_assert(equals(max_id(cl::sycl::range<1> { 3 }), cl::sycl::id<1> { 2 }), "max_id");
	static_assert(equals(max_id(cl::sycl::range<2> { 1, 1 }), cl::sycl::id<2> { 0, 0 }), "max_id");

	static_assert(equals(distance(id<1>{0}, id<1>{2}), range<1>{ 2 }), "distance");
	static_assert(equals(distance(id<1>{1}, id<1>{2}), range<1>{ 1 }), "distance");
	static_assert(equals(distance(id<2>{1, 3}, id<2>{2, 5}), range<2>{ 1, 2 }), "distance");
	static_assert(equals(distance(id<2>{1, 3}, id<2>{2, 1}), range<2>{ 1, -2 }), "distance");
	static_assert(equals(distance(id<2>{1, 3}, id<2>{2, 3}), range<2>{ 1, 0 }), "distance");*/
}

int main(int, char*[]) {

	sequence_static_assertions();
	iterator_static_assertions();

	sequence_examples();

	sycl_helper_assertions();

	cout << endl;
	cin.get();

	return 0;
}