#ifndef ITERATOR_H
#define ITERATOR_H

#include "celerity.h"
#include <stdexcept>
#include <cassert>

namespace algorithm
{
	template<typename T, size_t Dims>
	class iterator;

	template<typename T>
	class iterator<T, 1>
	{
	public:
		iterator(int pos, buffer<T, 1> & buffer)
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

		int operator*() const { return pos_; }
		buffer<T, 1> & buffer() const { return buffer_; }

	private:
		int pos_ = 0;
		::buffer<T, 1>& buffer_;
	};

	enum class iterator_type
	{
		one_to_one,
		neighbor,
	};

	template<typename T, size_t Dims, iterator_type Type, access_mode Mode>
	struct iterator_wrapper
	{
		iterator<T, Dims> iterator;
	};

	template<template <typename, size_t, iterator_type, access_mode> typename It,
		typename T, size_t Dims, iterator_type Type, access_mode Mode>
		auto get_access(handler cgh, It<T, Dims, Type, Mode> beg, It<T, Dims, Type, Mode> end)
	{
		assert(&beg.iterator.buffer() == &end.iterator.buffer());
		assert(*beg.iterator < *end.iterator);

		return beg.iterator.buffer().get_access<Mode>(cgh, range<1>{ *end.iterator - *beg.iterator });
	}

	template<typename T>
	iterator<T, 1> begin(buffer<T, 1> & buffer)
	{
		return iterator<T, 1>(0, buffer);
	}

	template<typename T>
	iterator<T, 1> end(buffer<T, 1> & buffer)
	{
		return iterator<T, 1>(buffer.size(), buffer);
	}

	template<access_mode Mode, typename T>
	auto one_to_one(const iterator<T, 1>& it)
	{
		return iterator_wrapper<T, 1, iterator_type::one_to_one, Mode>{ it };
	}

	template<access_mode Mode, typename T>
	auto neighbor(const iterator<T, 1> & it, int, int)
	{
		return iterator_wrapper<T, 1, iterator_type::neighbor, Mode>{ it };
	}

}

#endif