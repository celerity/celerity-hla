#ifndef ITERATOR_H
#define ITERATOR_H

#include "celerity.h"
#include "sycl.h"

namespace celerity::algorithm
{
	template<size_t Rank>
	class iterator
	{
	public:
		iterator(cl::sycl::id<Rank> pos, cl::sycl::range<Rank>& range)
			: pos_(pos),
			range_(range)
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
			pos_ = celerity::next(pos_, range_);

			if (pos_[0] != range_[0])
				return *this;

			for (auto i = 0; i < Rank; ++i)
				pos_[i] = range_[i];

			return *this;
		}

		[[nodiscard]] cl::sycl::id<Rank> operator*() const { return pos_; }

	private:
		cl::sycl::id<Rank> pos_ = 0;
		cl::sycl::range<Rank> range_;
	};
	
	template<typename T, size_t Rank>
	class buffer_iterator : public iterator<Rank>
	{
	public:
		buffer_iterator(cl::sycl::id<Rank> pos, buffer<T, Rank>& buffer)
			: iterator(pos, buffer.size()), buffer_(buffer)
		{
		}

		[[nodiscard]] buffer<T, Rank>& buffer() const { return buffer_; }

	private:
		celerity::buffer<T, Rank>& buffer_;
	};


	template<typename T, size_t Rank>
	cl::sycl::range<Rank> distance(buffer_iterator<T, Rank> from, buffer_iterator<T, Rank> to)
	{
		return celerity::distance(*from, *to);
	}
}

namespace celerity
{
	template<typename T, size_t Rank>
	algorithm::buffer_iterator<T, Rank> begin(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::buffer_iterator<T, Rank>(cl::sycl::id<Rank>{}, buffer);
	}

	template<typename T, size_t Rank>
	algorithm::buffer_iterator<T, Rank> end(celerity::buffer<T, Rank> & buffer)
	{
		return algorithm::buffer_iterator<T, Rank>(buffer.size(), buffer);
	}

	template<size_t Rank, typename Iterator, typename F>
	void for_each_index(Iterator beg, Iterator end, cl::sycl::range<Rank> r, const F& f)
	{
		std::for_each(algorithm::iterator<Rank>{ *beg, r }, algorithm::iterator<Rank>{ *end, r },
			[&](auto i)
			{
				f(cl::sycl::item<Rank>{ r, i });
			});
	}
}

#endif