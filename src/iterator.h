#ifndef ITERATOR_H
#define ITERATOR_H

#include "celerity.h"
#include "sycl_helper.h"
#include <stdexcept>
#include <cassert>

namespace celerity::algorithm
{
	template<typename T, size_t Rank>
	class iterator
	{
	public:
		iterator(cl::sycl::id<Rank> pos, celerity::buffer<T, Rank> & buffer)
			: pos_(pos),
			buffer_(buffer)
		{
		}

		bool operator ==(const iterator& rhs)
		{
			return equals(pos_, rhs.pos_);
		}

		bool operator !=(const iterator& rhs)
		{
			return !equals(pos_, rhs.pos_);
		}

		iterator& operator++()
		{
			pos_ = celerity::next(pos_, buffer_.size()); 			
			
			return *this;
		}

		[[nodiscard]] cl::sycl::id<Rank> operator*() const { return pos_; }
		[[nodiscard]] celerity::buffer<T, Rank> & buffer() const { return buffer_; }

	private:
		cl::sycl::id<Rank> pos_ = 0;
		celerity::buffer<T, Rank> & buffer_;
	};

	template<size_t Rank>
	class iterator_wod
	{
	public:
		iterator_wod(cl::sycl::id<Rank> pos, cl::sycl::range<Rank>& range)
			: pos_(pos),
			range_(range)
		{
		}

		bool operator ==(const iterator_wod& rhs)
		{
			return equals(pos_, rhs.pos_);
		}

		bool operator !=(const iterator_wod& rhs)
		{
			return !equals(pos_, rhs.pos_);
		}

		iterator_wod& operator++()
		{
			pos_ = celerity::next(pos_, range_);

			if (pos_[0] != range_[0])
				return *this;

			for(int i = 0; i < Rank; ++i)
				pos_ [i] = range_[i];

			return *this;
		}

		[[nodiscard]] cl::sycl::id<Rank> operator*() const { return pos_; }

	private:
		cl::sycl::id<Rank> pos_ = 0;
		cl::sycl::range<Rank> range_;
	};

	template<typename T, size_t Rank>
	cl::sycl::range<Rank> distance(iterator<T, Rank> from, iterator<T, Rank> to)
	{
		return celerity::distance(*from, *to);
	}
}

namespace celerity
{
	template<typename T, size_t Rank>
	algorithm::iterator<T, Rank> begin(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::iterator<T, Rank>(cl::sycl::id<Rank>{}, buffer);
	}

	template<typename T, size_t Rank>
	algorithm::iterator<T, Rank> end(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::iterator<T, Rank>(buffer.size(), buffer);
	}

	template<size_t Rank, typename Iterator, typename F>
	void for_each_index(Iterator beg, Iterator end, cl::sycl::range<Rank> r, const F& f)
	{
		std::for_each(algorithm::iterator_wod<Rank>{ *beg, r }, algorithm::iterator_wod<Rank>{ *end, r },
			[&](auto i)
			{
				f(cl::sycl::item<Rank>{ r, i });
			});
	}
}

#endif