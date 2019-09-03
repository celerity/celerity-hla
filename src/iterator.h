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
		iterator(cl::sycl::id<Rank> pos, celerity::buffer<T, 1> & buffer)
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
			pos_ = next(pos_, buffer_.size()); return *this;
		}

		[[nodiscard]] cl::sycl::id<Rank> operator*() const { return pos_; }
		[[nodiscard]] celerity::buffer<T, Rank> & buffer() const { return buffer_; }

	private:
		cl::sycl::id<Rank> pos_ = 0;
		celerity::buffer<T, 1> & buffer_;
	};

	template<typename T, size_t Rank>
	cl::sycl::id<Rank> distance(iterator<T, Rank> from, iterator<T, Rank> to)
	{
		return celerity::distance(*from, *to);
	}
}

namespace celerity
{
	template<typename T, size_t Rank>
	algorithm::iterator<T, Rank> begin(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::iterator<T, Rank>(cl::sycl::id<1>{0}, buffer);
	}

	template<typename T, size_t Rank>
	algorithm::iterator<T, Rank> end(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::iterator<T, Rank>(next(max_id(buffer.size()), buffer.size()), buffer);
	}
}

#endif