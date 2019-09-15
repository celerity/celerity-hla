// ReSharper disable once CppMissingIncludeGuard
#ifndef MOCK_CELERITY
#include <celerity>
#else
#ifndef CELERITY_H
#define CELERITY_H

#include <iostream>
#include <iterator>
#include <vector>
#include <array>

#include "iterator.h"
#include "sycl.h"

namespace celerity
{
struct runtime
{
	template <typename... Args>
	static void init(Args...) {}
};

struct handler
{
	int invocations;

	template <typename KernelName, int Rank, typename F>
	void parallel_for(cl::sycl::range<Rank> r, cl::sycl::id<Rank> offset, F f)
	{
		cl::sycl::id<Rank> end;

		for (int i = 0; i < Rank; ++i)
			end[i] = r[i];

		for_each_index(algorithm::iterator<Rank>{{}, r}, {end, r}, r, offset, f);
	}

	template <typename F>
	void run(F f)
	{
		f();
	}
};

class distr_queue
{
public:
	template <typename F>
	void submit(F f)
	{
		f(handler{++invocation_count_});
	}

	template <typename F>
	void with_master_access(F f)
	{
		f(handler{++invocation_count_});
	}

	void wait();

private:
	int invocation_count_ = 0;
};

enum class access_mode
{
	read,
	write,
	read_write
};

inline std::string to_string(const access_mode mode)
{
	switch (mode)
	{
	case access_mode::read:
		return " read";
	case access_mode::write:
		return "write";
	case access_mode::read_write:
		return "read_write";
	default:
		return "unknown";
	}
}

template <typename T, int Rank>
class buffer;

template <access_mode Mode, typename T, int Rank>
class accessor
{
public:
	explicit accessor(buffer<T, Rank> &buffer)
		: buffer_(buffer) {}

	decltype(auto) operator[](cl::sycl::item<Rank> item)
	{
		const auto idx = item.get_linear_id();
		decltype(auto) val = buffer_.data()[idx];
		print_access(item, val);
		return val;
	}

	T operator[](cl::sycl::item<Rank> item) const
	{
		const auto idx = item.get_linear_id();
		decltype(auto) val = buffer_.data()[idx];
		print_access(item, val);
		return val;
	}

	const T *get_pointer() const { return buffer_.data().data(); }

private:
	buffer<T, Rank> &buffer_;

	static void print_access(cl::sycl::item<Rank> idx, const T &value)
	{
#ifdef DEBUG_
		std::cout << typeid(T).name() << " ";
		std::cout << "accessor<" << to_string(Mode) << ", " << typeid(T).name() << ", " << Rank << ">";
		std::cout << "::operator [](";

		const auto id = idx.get_id();
		std::copy(begin(id), end(id), std::ostream_iterator<size_t>{std::cout, ","});

		std::cout << ") -> " << value << std::endl;
#endif
	}
};

template <typename T, int Rank>
struct buffer_type
{
	using value_type = T;
	static constexpr auto rank = Rank;
};

template <typename T, int Rank>
class buffer
{
public:
	using value_type = T;
	static constexpr auto rank = Rank;

	explicit buffer(cl::sycl::range<Rank> size)
		: buf_(size.size()), size_(size)
	{
	}

	buffer(const T *data, cl::sycl::range<Rank> size)
		: buffer(size)
	{
		if (!data)
			return;

		std::memcpy(buf_.data(), data, buf_.size() * sizeof(T));
	}

	template <access_mode mode, typename Rmt>
	auto get_access(handler cgh, cl::sycl::range<Rank> range, Rmt rm) { return accessor<mode, T, Rank>{*this}; }

	constexpr buffer_type<T, Rank> type() const
	{
		return buffer_type<T, Rank>();
	}

	[[nodiscard]] cl::sycl::range<Rank> size() const { return size_; }

	auto &data()
	{
		return buf_;
	}

private:
	std::vector<T> buf_;
	cl::sycl::range<Rank> size_;
};
} // namespace celerity

#endif
#endif