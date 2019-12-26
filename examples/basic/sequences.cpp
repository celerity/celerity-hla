#define _ITERATOR_DEBUG_LEVEL 0

#include "../../src/sequence.h"
#include "../../src/actions.h"
#include "../../src/task_sequence.h"
#include "../../src/kernel_traits.h"
#include "../../src/algorithm.h"

#include <iostream>

using namespace std;

void sequence_static_assertions()
{
	using namespace celerity;
	using namespace algorithm;

	auto hello_world = []() { std::cout << "hello world" << endl; };
	auto zero = [](handler&) {};

	using hello_world_t = decltype(hello_world);
	using zero_t = decltype(zero);

	static_assert(!algorithm::detail::has_call_operator_v<int>, "no call operator");
	static_assert(algorithm::detail::has_call_operator_v<decltype(zero)>, "no call operator");
	static_assert(algorithm::detail::has_call_operator_v<decltype(hello_world)>, "no call operator");
	static_assert(algorithm::detail::get_accessor_type<decltype(zero), 0>() == access_type::one_to_one, "get_accessor_type");
	//static_assert(algorithm::detail::get_accessor_type<algorithm::buffer_iterator<float, 1>, 0>() == access_type::invalid, "get_accessor_type");

	static_assert(!is_contiguous_iterator<decltype(begin(std::declval<vector<float>>()))>());
	static_assert(is_contiguous_iterator<decltype(std::declval<vector<float>>().data())>());
}

void sequence_examples()
{
	using namespace celerity;
	using namespace algorithm;
	using namespace actions;

	using namespace algorithm;

	buffer<float, 1> b{{5}};
	buffer<float, 1> c{{5}};
	buffer<float, 1> b_out{{5}};

	distr_queue q;

	//algorithm::fill<class fill>(q, begin(b), end(b), 1.f);
	algorithm::transform<class mul2>(q, next(begin(b)), end(b), begin(c), [](const float x) { return 2 * x; });

	/*algorithm::transform(algorithm::distr<class sum>(q), begin(b), end(b), begin(b_out),
						 [&](slice<float, 0> x) {
							 auto sum = *x;

							 for (int i = 0; i < 5; ++i)
								 sum += x[{i}];

							 return sum;
						 });*/

	algorithm::transform(master(q), begin(b), end(b), begin(b_out),
						 [](float x) {
							 return 2 * x;
						 });

	/*algorithm::transform(master(q), begin(b), end(b), begin(b_out),
		[](chunk<float, 2> x)
		{
			return 2 * x[{0}];
		});*/

	algorithm::transform(algorithm::distr<class product>(q), begin(b), end(b), begin(c), begin(b_out),
						 [](float x, float y) {
							 return x * y;
						 });

	algorithm::transform(algorithm::master(q), begin(b), end(b), begin(c), begin(b_out),
						 [](float x, float y) {
							 return x * y;
						 });

	algorithm::transform(algorithm::master(q), begin(b), end(b), begin(b_out),
						 [](cl::sycl::item<1> item, float x) {
							 return x;
						 });
}

void sycl_helper_assertions()
{
	using namespace celerity;
	using namespace cl::sycl;

	assert(count(cl::sycl::range<1> { 2 }) == 2 && "count");
	assert(count(cl::sycl::range<1> { 3 }) == 3 && "count");
	assert(count(cl::sycl::range<2> { 1, 2 }) == 2 && "count");
	assert(count(cl::sycl::range<2> { 2, 2 }) == 4 && "count");
	assert(count(cl::sycl::range<3> { 2, 2, 2 }) == 8 && "count");
	assert(count(cl::sycl::range<3> { 2, 3, 2 }) == 12 && "count");

	assert(equals(next(cl::sycl::id<1> { 0 }, cl::sycl::range<1> { 2 }), cl::sycl::id<1> { 1 }) && "next");
	assert(equals(next(cl::sycl::id<1> { 1 }, cl::sycl::range<1> { 2 }), cl::sycl::id<1> { 2 }) && "next");
	assert(equals(next(cl::sycl::id<1> { 2 }, cl::sycl::range<1> { 2 }), cl::sycl::id<1> { 3 }) && "next");
	assert(equals(next(cl::sycl::id<2> { 0, 0 }, cl::sycl::range<2> { 2, 2 }), cl::sycl::id<2> { 0, 1 }) && "next");
	assert(equals(next(cl::sycl::id<2> { 0, 0 }, cl::sycl::range<2> { 2, 1 }), cl::sycl::id<2> { 1, 0 }) && "next");
	assert(equals(next(cl::sycl::id<2> { 0, 1 }, cl::sycl::range<2> { 2, 2 }), cl::sycl::id<2> { 1, 0 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 1 }, cl::sycl::range<3> { 2, 2, 2 }), cl::sycl::id<3> { 0, 1, 0 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 1, 1 }, cl::sycl::range<3> { 2, 2, 2 }), cl::sycl::id<3> { 1, 0, 0 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, cl::sycl::range<3> { 2, 2, 2 }, 2), cl::sycl::id<3> { 0, 1, 0 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, cl::sycl::range<3> { 2, 2, 2 }, 3), cl::sycl::id<3> { 0, 1, 1 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, cl::sycl::range<3> { 2, 2, 2 }, 4), cl::sycl::id<3> { 1, 0, 0 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, cl::sycl::range<3> { 2, 2, 2 }, 7), cl::sycl::id<3> { 1, 1, 1 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 0, 0, 0 }, cl::sycl::range<3> { 3, 3, 3 }, 7), cl::sycl::id<3> { 0, 2, 1 }) && "next");
	assert(equals(next(cl::sycl::id<3> { 2, 2, 2 }, cl::sycl::range<3> { 3, 3, 3 }, 1), cl::sycl::id<3> { 3, 0, 0 }) && "next");
	
	assert(equals(prev(cl::sycl::id<1> { 2 }, max_id(cl::sycl::range<1> { 2 })), cl::sycl::id<1> { 1 }) && "prev");
	assert(equals(prev(cl::sycl::id<2> { 0, 1 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 0, 0 }) && "prev");
	assert(equals(prev(cl::sycl::id<2> { 1, 0 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 0, 1 }) && "prev");
	assert(equals(prev(cl::sycl::id<2> { 1, 1 }, max_id(cl::sycl::range<2> { 2, 2 })), cl::sycl::id<2> { 1, 0 }) && "prev");
	assert(equals(prev(cl::sycl::id<3> { 2, 2, 0 }, max_id(cl::sycl::range<3> { 3, 3, 3 })), cl::sycl::id<3> { 2, 1, 2 }) && "prev");
	assert(equals(prev(cl::sycl::id<3> { 2, 0, 0 }, max_id(cl::sycl::range<3> { 3, 3, 3 })), cl::sycl::id<3> { 1, 2, 2 }) && "prev");
	assert(equals(prev(cl::sycl::id<3> { 2, 2, 3 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 2), cl::sycl::id<3> { 2, 2, 1 }) && "prev");
	assert(equals(prev(cl::sycl::id<3> { 2, 2, 2 }, max_id(cl::sycl::range<3> { 3, 3, 3 }), 3), cl::sycl::id<3> {2, 1, 2 }) && "prev");

	assert(equals(max_id(cl::sycl::range<3> { 2, 2, 2 }), cl::sycl::id<3> { 1, 1, 1 }) && "max_id");
	assert(equals(max_id(cl::sycl::range<1> { 3 }), cl::sycl::id<1> { 2 }) && "max_id");
	assert(equals(max_id(cl::sycl::range<2> { 1, 1 }), cl::sycl::id<2> { 0, 0 }) && "max_id");

	assert(equals(distance(cl::sycl::id<1>{0}, cl::sycl::id<1>{2}), cl::sycl::range<1>{ 2 }) && "distance");
	assert(equals(distance(cl::sycl::id<1>{1}, cl::sycl::id<1>{2}), cl::sycl::range<1>{ 1 }) && "distance");
	assert(equals(distance(cl::sycl::id<2>{1, 3}, cl::sycl::id<2>{2, 5}), cl::sycl::range<2>{ 1, 2 }) && "distance");
	//assert(equals(distance(cl::sycl::id<2>{1, 3}, cl::sycl::id<2>{2, 1}), cl::sycl::range<2>{ 1, -2 }) && "distance");
	assert(equals(distance(cl::sycl::id<2>{1, 3}, cl::sycl::id<2>{2, 3}), cl::sycl::range<2>{ 1, 0 }) && "distance");
}

int main(int, char *[])
{
	sequence_static_assertions();

	sequence_examples();

	sycl_helper_assertions();

	cout << endl;
	cin.get();

	return 0;
}