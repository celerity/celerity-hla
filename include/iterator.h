#ifndef ITERATOR_H
#define ITERATOR_H

#include "sycl.h"
#include <type_traits>

namespace celerity
{
	template <typename T, int Rank>
	class buffer;
}

namespace celerity::algorithm
{
	template <int Rank>
	class iterator
	{
	public:
		iterator(cl::sycl::id<Rank> pos, cl::sycl::range<Rank> range)
			: pos_(pos),
			  range_(range)
		{
			stride_[Rank - 1] = 1;
		}

		iterator(cl::sycl::id<Rank> pos, cl::sycl::range<Rank> range, cl::sycl::id<Rank> stride)
			: pos_(pos),
			  range_(range),
			  stride_(stride)
		{
		}

		bool operator==(const iterator &rhs)
		{
			return detail::equals(pos_, rhs.pos_);
		}

		bool operator!=(const iterator &rhs)
		{
			return !detail::equals(pos_, rhs.pos_);
		}

		iterator &operator++()
		{
			return operator+=(stride_);
		}

		iterator &operator+=(cl::sycl::id<Rank> offset)
		{
			pos_ = next(pos_, range_, offset);

			if (pos_[0] < range_[0])
				return *this;

			saturate();

			return *this;
		}

		[[nodiscard]] cl::sycl::id<Rank> operator*() const { return pos_; }

		void set_stride(cl::sycl::id<Rank> stride) { stride_ = stride; }
		void set_range(cl::sycl::range<Rank> range)
		{
			range_ = range;
			saturate();
		}
		void set_pos(cl::sycl::id<Rank> pos)
		{
			pos_ = pos;
			saturate();
		}

	private:
		cl::sycl::id<Rank> pos_ = 0;
		cl::sycl::range<Rank> range_;
		cl::sycl::id<Rank> stride_{};

		void saturate()
		{
			for (auto i = 0; i < Rank; ++i)
			{
				pos_[i] = std::min(pos_[i], range_[i]);
				pos_[i] = std::max(pos_[i], size_t(0));
			}
		}
	};

	namespace detail
	{
		struct celerity_iterator_tag
		{
		}; // : contiguous_iterator_tag
	}	   // namespace detail

	template <typename T, int Rank>
	class buffer_iterator : public iterator<Rank>
	{
	public:
		using iterator_category = detail::celerity_iterator_tag;
		using value_type = T;
		using difference_type = long;
		using pointer = std::add_pointer_t<T>;
		using reference = std::add_lvalue_reference_t<T>;

		buffer_iterator(cl::sycl::id<Rank> pos, buffer<T, Rank> buffer)
			: iterator<Rank>(pos, buffer.get_range()), buffer_(buffer)
		{
		}

		buffer_iterator &operator++()
		{
			iterator<Rank>::operator++();
			return *this;
		}

		[[nodiscard]] buffer<T, Rank> get_buffer() const { return buffer_; }

	private:
		celerity::buffer<T, Rank> buffer_;
	};

	namespace detail
	{

		template <int Rank>
		cl::sycl::range<Rank> distance(iterator<Rank> from, iterator<Rank> to)
		{
			return distance(*from, *to);
		}

		template <typename T, int Rank>
		cl::sycl::range<Rank> distance(buffer_iterator<T, Rank> from, buffer_iterator<T, Rank> to)
		{
			return distance(*from, *to);
		}

		template <int Rank, typename Iterator, typename F>
		void for_each_index(Iterator beg, Iterator end, cl::sycl::range<Rank> r, cl::sycl::id<Rank> offset, const F &f)
		{
			std::for_each(algorithm::iterator<Rank>{*beg, r}, algorithm::iterator<Rank>{*end, r},
						  [&](auto i) {
							  f(cl::sycl::detail::make_item(i + offset, r, offset));
						  });
		}

	} // namespace detail

	namespace traits
	{

		template <typename T>
		constexpr bool is_celerity_iterator_v = std::is_same_v<detail::celerity_iterator_tag, typename std::iterator_traits<T>::iterator_category>;

		template <typename T>
		constexpr bool is_contiguous_iterator()
		{
			// we can only detect raw pointers for now
			using value_t = typename std::iterator_traits<T>::value_type;
			using pointer_t = typename std::iterator_traits<T>::pointer;
			using iterator_t = std::decay_t<T>;

			return std::is_trivially_copyable_v<value_t> &&
				   std::is_pointer_v<iterator_t> &&
				   std::is_same_v<pointer_t, iterator_t>;
		}

	} // namespace traits

	template <int Rank>
	iterator<Rank> next(iterator<Rank> it)
	{
		return ++it;
	}

	template <typename T, int Rank>
	buffer_iterator<T, Rank> next(buffer_iterator<T, Rank> it)
	{
		return ++it;
	}

} // namespace celerity::algorithm

namespace celerity
{
	template <typename T, int Rank>
	algorithm::buffer_iterator<T, Rank> begin(celerity::buffer<T, Rank> buffer)
	{
		return algorithm::buffer_iterator<T, Rank>(cl::sycl::id<Rank>{}, buffer);
	}

	template <typename T, int Rank>
	algorithm::buffer_iterator<T, Rank> end(celerity::buffer<T, Rank> buffer)
	{
		return algorithm::buffer_iterator<T, Rank>(buffer.get_range(), buffer);
	}

} // namespace celerity

#endif