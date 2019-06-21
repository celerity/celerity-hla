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

namespace cl::sycl
{
	template<size_t Rank>
	using range = std::array<int, Rank>;

	template<size_t Rank>
	using item = std::array<int, Rank>;
}

namespace celerity
{
	namespace detail
	{
		template<size_t Rank, size_t...Is>
		int dispatch_count(cl::sycl::range<Rank> r, std::index_sequence<Is...>)
		{
			return (std::get<Is>(r) * ... * 1);
		}
	}

	template<size_t Rank>
	int count(cl::sycl::range<Rank> r)
	{
		return detail::dispatch_count(r, std::make_index_sequence<Rank>{});
	}

	struct handler
	{
		int invocations;

		template<typename KernelName, size_t Rank, typename F>
		void parallel_for(cl::sycl::range<Rank> r, F f)
		{
			if constexpr (Rank == 1)
			{
				for (auto i = 0; i < count(r); ++i)
				{
					f(cl::sycl::item<Rank>{i});
				}
			}
			else
			{
				throw std::logic_error("not implemented");
			}
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

	template<access_mode Mode, typename T, size_t Rank>
	struct accessor
	{
		T& operator[](cl::sycl::item<Rank> idx)
		{
			std::cout << typeid(T).name() << "& ";
			print_accessor_type();
			std::cout << "::operator [](";
			std::copy(begin(idx), idx.end(), std::ostream_iterator<int>{ std::cout, "," });
			std::cout << ")" << std::endl;

			static T x{};
			return x;
		}

		T operator[](cl::sycl::item<Rank> idx) const
		{
			std::cout << typeid(T).name() << "  ";
			print_accessor_type();
			std::cout << "::operator [](";
			std::copy(idx.begin(), idx.end(), std::ostream_iterator<int>{ std::cout, "," });
			std::cout << ")" << std::endl;

			return T{};
		}

		static void print_accessor_type()
		{
			std::cout << "accessor<" << to_string(Mode) << ", " << typeid(T).name() << ", " << Rank << ">";
		}
	};

	template<typename T, size_t Rank>
	class buffer
	{
	public:
		explicit buffer(cl::sycl::range<Rank> size)
			: buf_(count(size))
		{
		}

		template<access_mode mode>
		auto get_access(handler cgh, cl::sycl::range<Rank> range) { return accessor<mode, T, Rank>{}; }

		[[nodiscard]]
		size_t size() const { return buf_.size(); }

	private:
		std::vector<T> buf_;
	};
}

#endif
#endif