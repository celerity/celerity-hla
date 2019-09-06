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

#include "sycl_helper.h"

namespace celerity
{
	struct handler
	{
		int invocations;

		template<typename KernelName, size_t Rank, typename F>
		void parallel_for(cl::sycl::range<Rank> r, F f)
		{
			cl::sycl::id<Rank> end;
			
			for (int i = 0; i < Rank; ++i)
				end[i] = r[i];

			for_each_index(algorithm::iterator<Rank>{ { }, r }, { end, r }, r, f);
		}

		template<typename F>
		void run(F f)
		{
			f();
		}
	};

	class distr_queue
	{
	public:
		template<typename F>
		void submit(F f)
		{
			f(handler{ ++invocation_count_ });
		}

		template<typename F>
		void with_master_access(F f)
		{
			f(handler{ ++invocation_count_ });
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
		case access_mode::read: return "read";
		case access_mode::write: return "write";
		case access_mode::read_write: return "read_write";
		default: return "unknown";
		}
	}

	template<typename T, size_t Rank>
	class buffer;

	template<access_mode Mode, typename T, size_t Rank>
	class accessor
	{
	public:
		explicit accessor(buffer<T, Rank>& buffer)
			: buffer_(buffer) {}

		decltype(auto) operator[](cl::sycl::item<Rank> idx)
		{
			std::cout << typeid(T).name() << "& ";
			print_accessor_type();
			std::cout << "::operator [](";

			const auto id = idx.get_id();

			std::copy(begin(id), end(id), std::ostream_iterator<int>{ std::cout, "," });
			std::cout << ")" << std::endl;

			//static_assert(Rank == 1);

			return buffer_.data()[linearize(idx.get_id(), buffer_.size())[0]];
		}

		T operator[](cl::sycl::item<Rank> idx) const
		{
			std::cout << typeid(T).name() << "  ";
			print_accessor_type();
			std::cout << "::operator [](";

			const auto id = idx.get_id();

			std::copy(begin(id), end(id), std::ostream_iterator<int>{ std::cout, "," });

			std::cout << ")" << std::endl;

			static_assert(Rank == 1);

			return buffer_.data()[linearize(idx.get_id(), buffer_.size())[0]];
		}

		static void print_accessor_type()
		{
			std::cout << "accessor<" << to_string(Mode) << ", " << typeid(T).name() << ", " << Rank << ">";
		}

	private:
		buffer<T, Rank>& buffer_;
	};

	template<typename T, size_t Rank>
	struct buffer_type
	{
		using value_type = T;
		static constexpr auto rank = Rank;
	};

	template<typename T, size_t Rank>
	class buffer
	{
	public:
		using value_type = T;
		static constexpr auto rank = Rank;

		explicit buffer(cl::sycl::range<Rank> size)
			: buf_(count(size)), size_(size)
		{
		}

		buffer(const T* data, cl::sycl::range<Rank> size)
			: buffer(size)
		{
			std::memcpy(buf_.data(), data, buf_.size() * sizeof(T));
		}

		template<access_mode mode>
		auto get_access(handler cgh, cl::sycl::range<Rank> range) { return accessor<mode, T, Rank>{*this}; }

		constexpr buffer_type<T, Rank> type() const {
			return buffer_type<T, Rank>();
		}

		[[nodiscard]]
		cl::sycl::range<Rank> size() const { return size_; }

		auto& data() { return buf_; }

	private:
		std::vector<T> buf_;
		cl::sycl::range<Rank> size_;
	};
}

#endif
#endif