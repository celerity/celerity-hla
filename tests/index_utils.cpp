#define CATCH_CONFIG_MAIN
#include "../src/catch/catch.hpp"
#include "../src/sycl.h"

using namespace celerity;
using namespace cl::sycl;

SCENARIO("ranges can be counted", "[cl::sycl::range]")
{
	GIVEN("A one-dimensional range")
	{
		REQUIRE(count(cl::sycl::range<1>{2}) == 2);
		REQUIRE(count(cl::sycl::range<1>{3}) == 3);
	}

	GIVEN("A two-dimensional range")
	{
		REQUIRE(count(cl::sycl::range<2>{1, 2}) == 2);
		REQUIRE(count(cl::sycl::range<2>{2, 2}) == 4);
	}

	GIVEN("A three-dimensional range")
	{
		REQUIRE(count(cl::sycl::range<3>{2, 2, 2}) == 8);
		REQUIRE(count(cl::sycl::range<3>{2, 3, 2}) == 12);
	}
}

SCENARIO("ids can be incremented", "[cl::sycl::id]")
{
	GIVEN("A one-dimensional id")
	{
		REQUIRE(equals(next(cl::sycl::id<1>{0}, cl::sycl::range<1>{2}), cl::sycl::id<1>{1}));
		REQUIRE(equals(next(cl::sycl::id<1>{1}, cl::sycl::range<1>{2}), cl::sycl::id<1>{2}));
		REQUIRE(equals(next(cl::sycl::id<1>{2}, cl::sycl::range<1>{2}), cl::sycl::id<1>{3}));
	}

	GIVEN("A two-dimensional id")
	{
		REQUIRE(equals(next(cl::sycl::id<2>{0, 0}, cl::sycl::range<2>{2, 2}), cl::sycl::id<2>{0, 1}));
		REQUIRE(equals(next(cl::sycl::id<2>{0, 0}, cl::sycl::range<2>{2, 1}), cl::sycl::id<2>{1, 0}));
		REQUIRE(equals(next(cl::sycl::id<2>{0, 1}, cl::sycl::range<2>{2, 2}), cl::sycl::id<2>{1, 0}));
	}

	GIVEN("A three-dimensional id")
	{
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 1}, cl::sycl::range<3>{2, 2, 2}), cl::sycl::id<3>{0, 1, 0}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 1, 1}, cl::sycl::range<3>{2, 2, 2}), cl::sycl::id<3>{1, 0, 0}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 0}, cl::sycl::range<3>{2, 2, 2}, 2), cl::sycl::id<3>{0, 1, 0}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 0}, cl::sycl::range<3>{2, 2, 2}, 3), cl::sycl::id<3>{0, 1, 1}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 0}, cl::sycl::range<3>{2, 2, 2}, 4), cl::sycl::id<3>{1, 0, 0}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 0}, cl::sycl::range<3>{2, 2, 2}, 7), cl::sycl::id<3>{1, 1, 1}));
		REQUIRE(equals(next(cl::sycl::id<3>{0, 0, 0}, cl::sycl::range<3>{3, 3, 3}, 7), cl::sycl::id<3>{0, 2, 1}));
		REQUIRE(equals(next(cl::sycl::id<3>{2, 2, 2}, cl::sycl::range<3>{3, 3, 3}, 1), cl::sycl::id<3>{3, 0, 0}));
	}
}

SCENARIO("ids can be decremented", "[cl::sycl::id]")
{
	GIVEN("A one-dimensional id")
	{
		REQUIRE(equals(prev(cl::sycl::id<1>{2}, cl::sycl::range<1>{2}), cl::sycl::id<1>{1}));
	}

	GIVEN("A two-dimensional id")
	{
		REQUIRE(equals(prev(cl::sycl::id<2>{0, 1}, cl::sycl::range<2>{2, 2}), cl::sycl::id<2>{0, 0}));
		REQUIRE(equals(prev(cl::sycl::id<2>{1, 0}, cl::sycl::range<2>{2, 2}), cl::sycl::id<2>{0, 1}));
		REQUIRE(equals(prev(cl::sycl::id<2>{1, 1}, cl::sycl::range<2>{2, 2}), cl::sycl::id<2>{1, 0}));
	}

	GIVEN("A three-dimensional id")
	{
		REQUIRE(equals(prev(cl::sycl::id<3>{2, 2, 0}, cl::sycl::range<3>{3, 3, 3}), cl::sycl::id<3>{2, 1, 2}));
		REQUIRE(equals(prev(cl::sycl::id<3>{2, 0, 0}, cl::sycl::range<3>{3, 3, 3}), cl::sycl::id<3>{1, 2, 2}));
		REQUIRE(equals(prev(cl::sycl::id<3>{2, 2, 3}, cl::sycl::range<3>{3, 3, 3}, 2), cl::sycl::id<3>{2, 2, 1}));
		REQUIRE(equals(prev(cl::sycl::id<3>{2, 2, 2}, cl::sycl::range<3>{3, 3, 3}, 3), cl::sycl::id<3>{2, 1, 2}));
	}
}

SCENARIO("ranges have a maximum id", "[cl::sycl::range]")
{
	GIVEN("A one-dimensional range")
	{
		REQUIRE(equals(max_id(cl::sycl::range<1>{3}), cl::sycl::id<1>{2}));
	}

	GIVEN("A two-dimensional id")
	{
		REQUIRE(equals(max_id(cl::sycl::range<2>{1, 1}), cl::sycl::id<2>{0, 0}));
	}

	GIVEN("A three-dimensional id")
	{
		REQUIRE(equals(max_id(cl::sycl::range<3>{2, 2, 2}), cl::sycl::id<3>{1, 1, 1}));
	}
}

SCENARIO("distance between two ids can be calculated", "[cl::sycl::id]")
{
	GIVEN("One-dimensional ids")
	{
		REQUIRE(equals(distance(cl::sycl::id<1>{0}, cl::sycl::id<1>{2}), cl::sycl::range<1>{2}));
		REQUIRE(equals(distance(cl::sycl::id<1>{1}, cl::sycl::id<1>{2}), cl::sycl::range<1>{1}));
	}

	GIVEN("Two-dimensional ids")
	{
		REQUIRE(equals(distance(cl::sycl::id<2>{1, 3}, cl::sycl::id<2>{2, 5}), cl::sycl::range<2>{1, 2}));
		REQUIRE(equals(distance(cl::sycl::id<2>{1, 3}, cl::sycl::id<2>{2, 3}), cl::sycl::range<2>{1, 0}));
	}

	GIVEN("Three-dimensional ids")
	{
		REQUIRE(equals(distance(cl::sycl::id<3>{1, 3, 2}, cl::sycl::id<3>{2, 5, 5}), cl::sycl::range<3>{1, 2, 3}));
		REQUIRE(equals(distance(cl::sycl::id<3>{1, 3, 2}, cl::sycl::id<3>{2, 3, 3}), cl::sycl::range<3>{1, 0, 1}));
	}
}