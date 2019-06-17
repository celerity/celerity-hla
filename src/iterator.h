#ifndef ITERATOR_H
#define ITERATOR_H

#include "celerity.h"
#include <stdexcept>
#include <cassert>

namespace celerity::algorithm
{
	template<typename T, size_t Dims>
	class iterator;

	template<typename T>
	class iterator<T, 1>
	{
	public:
		iterator(int pos, celerity::buffer<T, 1> & buffer)
			: pos_(pos),
			buffer_(buffer)
		{
		}

		bool operator ==(const iterator& rhs)
		{
			return pos_ == rhs.pos_;
		}

		iterator& operator++()
		{
			pos_++; return *this;
		}

		[[nodiscard]] int operator*() const { return pos_; }
		[[nodiscard]] celerity::buffer<T, 1> & buffer() const { return buffer_; }

	private:
		int pos_ = 0;
		celerity::buffer<T, 1>& buffer_;
	};

	enum class iterator_type
	{
		one_to_one,
		neighbor,
		slice,
	};

	template<typename T, size_t Dims, iterator_type Type, celerity::access_mode Mode>
	struct iterator_wrapper
	{
		iterator<T, Dims> iterator;
	};

	template<celerity::access_mode Mode, typename T>
	auto one_to_one(const iterator<T, 1>& it)
	{
		return iterator_wrapper<T, 1, iterator_type::one_to_one, Mode>{ it };
	}

	template<celerity::access_mode Mode, typename T>
	auto neighbor(const iterator<T, 1> & it, int, int)
	{
		return iterator_wrapper<T, 1, iterator_type::neighbor, Mode>{ it };
	}

}

namespace celerity
{
	template<typename T>
	algorithm::iterator<T, 1> begin(celerity::buffer<T, 1> & buffer)
	{
		return algorithm::iterator<T, 1>(0, buffer);
	}

	template<typename T>
	algorithm::iterator<T, 1> end(celerity::buffer<T, 1> & buffer)
	{
		return algorithm::iterator<T, 1>(buffer.size(), buffer);
	}
}

#endif