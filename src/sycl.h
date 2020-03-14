#ifndef SYCL_HELPER_H
#define SYCL_HELPER_H

#include <utility>

#ifdef MOCK_CELERITY

#include "sycl/item.hpp"
#include "sycl/range.hpp"
#include "sycl/id.hpp"
#include "sycl/vec.hpp"

namespace cl::sycl
{
template <int Dimensions, bool with_offset = true>
using item = trisycl::item<Dimensions, with_offset>;

template <int Dimensions>
using id = trisycl::id<Dimensions>;

template <int Dimensions>
using range = trisycl::range<Dimensions>;

using float2 = trisycl::float2;
using float3 = trisycl::float3;

struct exception
{
	const char *what() { return nullptr; }
};
} // namespace cl::sycl

#else
#include <CL/sycl.hpp>
#endif

namespace std
{
template <int Idx, int Rank>
auto get(const cl::sycl::range<Rank> &r)
{
	return r[Idx];
}
} // namespace std

namespace cl::sycl
{
template <int Dimensions>
using rel_id = std::array<int, Dimensions>;
}

namespace celerity::algorithm
{
namespace detail
{
template <int Rank, size_t... Is>
constexpr size_t dispatch_count(cl::sycl::range<Rank> r, std::index_sequence<Is...>)
{
	return (std::get<Is>(r) * ... * 1);
}

template <int Rank, template <int> typename T, size_t... Is>
constexpr bool dispatch_equals(T<Rank> lhs, T<Rank> rhs, std::index_sequence<Is...>)
{
	static_assert(Rank > 0, "Rank must be a postive integer greater than zero");
	return ((lhs[Is] == rhs[Is]) && ...);
}

template <int Rank>
constexpr void incr(int idx, cl::sycl::id<Rank> &id, cl::sycl::range<Rank> r)
{
	while (id[idx + 1] >= r[idx + 1])
	{
		id[idx + 1] -= r[idx + 1];
		++id[idx];
	}
}

template <int Rank>
constexpr void decr_n(int idx, cl::sycl::id<Rank> &id, cl::sycl::id<Rank> r)
{
	while (id[idx + 1] > r[idx + 1])
	{
		id[idx + 1] += r[idx + 1] + 1;
		--id[idx];
	}
}

template <int Rank, size_t... Is>
constexpr void dispatch_next(cl::sycl::id<Rank> &id, cl::sycl::range<Rank> max_id, std::index_sequence<Is...>)
{
	static_assert(Rank > 0, "Rank must be a postive integer greater than zero");
	(incr(Rank - 2 - Is, id, max_id), ...);
}

template <int Rank, size_t... Is>
constexpr void dispatch_prev_n(cl::sycl::id<Rank> &id, cl::sycl::id<Rank> max_id, std::index_sequence<Is...>)
{
	static_assert(Rank > 0, "Rank must be a postive integer greater than zero");
	(decr_n(Rank - 2 - Is, id, max_id), ...);
}

cl::sycl::id<1> linearize(cl::sycl::id<1> idx, cl::sycl::range<1>)
{
	return idx;
}

cl::sycl::id<1> linearize(cl::sycl::id<2> idx, cl::sycl::range<2> r)
{
	return {idx[0] * r[0] + idx[1]};
}

cl::sycl::id<1> linearize(cl::sycl::id<3> idx, cl::sycl::range<3> r)
{
	return {idx[0] * r[1] * r[2] + idx[1] * r[2] + idx[2]};
}

template <int Rank>
constexpr cl::sycl::id<Rank> max_id(cl::sycl::range<Rank> r)
{
	cl::sycl::id<Rank> offset{};

	for (int i = 0; i < Rank; ++i)
	{
		offset[i] += r[i] - 1;
	}

	return offset;
}

template <int Rank>
constexpr cl::sycl::range<Rank> distance(cl::sycl::id<Rank> from, cl::sycl::id<Rank> to)
{
	cl::sycl::range<Rank> dist{};

	for (int i = 0; i < Rank; ++i)
	{
		dist[i] = to[i] - from[i];
	}

	return dist;
}

template <int Rank>
constexpr bool equals(cl::sycl::id<Rank> lhs, cl::sycl::id<Rank> rhs)
{
	return detail::dispatch_equals(lhs, rhs, std::make_index_sequence<Rank>{});
}

template <int Rank>
constexpr bool equals(cl::sycl::range<Rank> lhs, cl::sycl::range<Rank> rhs)
{
	return detail::dispatch_equals(lhs, rhs, std::make_index_sequence<Rank>{});
}

template <int Rank>
cl::sycl::id<1> linearize(cl::sycl::id<Rank> idx, cl::sycl::range<Rank> r)
{
	return detail::linearize(idx, r);
}

template <int Rank>
constexpr size_t count(cl::sycl::range<Rank> r)
{
	return detail::dispatch_count(r, std::make_index_sequence<Rank>{});
}

} // namespace detail

template <int Rank>
constexpr cl::sycl::id<Rank> next(cl::sycl::id<Rank> idx, cl::sycl::range<Rank> max_id, int distance = 1)
{
	cl::sycl::id<Rank> out = idx;

	out[Rank - 1] += distance;

	if constexpr (Rank > 1)
	{
		detail::dispatch_next(out, max_id, std::make_index_sequence<Rank - 1>{});
	}

	return out;
}

template <int Rank>
constexpr cl::sycl::id<Rank> prev(cl::sycl::id<Rank> idx, cl::sycl::range<Rank> r, int distance = 1)
{
	cl::sycl::id<Rank> out = idx;

	out[Rank - 1] -= distance;

	if constexpr (Rank > 1)
	{
		detail::dispatch_prev_n(out, detail::max_id(r), std::make_index_sequence<Rank - 1>{});
	}

	return out;
}

namespace detail
{

template <int Rank>
constexpr std::tuple<cl::sycl::id<Rank>, bool> try_next(cl::sycl::id<Rank> idx, cl::sycl::range<Rank> r, int distance = 1)
{
	cl::sycl::id<Rank> out = next(idx, r, distance);
	return {out, equal(out, idx)};
}

} // namespace detail

} // namespace celerity::algorithm

#endif // SYCL_HELPER_H