#ifndef ACCESSOR_PROXY_H
#define ACCESSOR_PROXY_H

#include "celerity_helper.h"

#include "accessors.h"
#include "accessor_traits.h"
#include "accessor_type.h"
#include "accessor_iterator.h"

#include "iterator.h"
#include "policy.h"
#include "item_context.h"

#include <type_traits>
#include <cmath>

namespace celerity::algorithm::detail
{

	template <typename AccessorType>
	class accessor_proxy_base
	{
	protected:
		accessor_proxy_base(AccessorType acc)
			: accessor_(acc) {}

	public:
		const auto &get_accessor() const { return accessor_; }

	private:
		AccessorType accessor_;
	};

	template <typename T, int Rank, typename AccessorType, typename Type>
	class accessor_proxy;

	template <typename T, int Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, one_to_one>
		: public accessor_proxy_base<AccessorType>
	{
	public:
		using base = accessor_proxy_base<AccessorType>;

		explicit accessor_proxy(AccessorType acc, cl::sycl::id<Rank>, cl::sycl::range<Rank>)
			: base(acc) {}

		decltype(auto) operator[](const cl::sycl::item<Rank> item) const
		{
			return base::get_accessor()[item];
		}

		decltype(auto) operator[](const cl::sycl::item<Rank> item)
		{
			return base::get_accessor()[item];
		}

		decltype(auto) operator[](item_shared_data<Rank, T> item) const
		{
			return base::get_accessor()[item];
		}

		decltype(auto) operator[](item_shared_data<Rank, T> item)
		{
			return base::get_accessor()[item];
		}
	};

	template <typename T, int Rank, typename AccessorType>
	class accessor_proxy<T, Rank, AccessorType, all<T, Rank>>
		: public accessor_proxy_base<AccessorType>
	{
	public:
		using base = accessor_proxy_base<AccessorType>;

		explicit accessor_proxy(AccessorType acc, cl::sycl::id<Rank> offset, cl::sycl::range<Rank> range)
			: base(acc), offset_(offset), range_(range) {}

		all<T, Rank> operator[](const cl::sycl::item<Rank>) const
		{
			return {base::get_accessor(), offset_, range_};
		}

	private:
		cl::sycl::id<Rank> offset_;
		cl::sycl::range<Rank> range_;
	};

	template <typename T, int Rank, typename AccessorType, size_t Dim,
			  bool Transposed>
	class accessor_proxy<T, Rank, AccessorType, slice<T, Dim, Transposed>>
		: public accessor_proxy_base<AccessorType>
	{
	public:
		using base = accessor_proxy_base<AccessorType>;

		static_assert(Dim >= 0 && Dim < Rank, "Dim out of bounds");

		explicit accessor_proxy(AccessorType acc, cl::sycl::id<Rank>, cl::sycl::range<Rank>)
			: base(acc) {}

		slice<T, Dim, Transposed> operator[](const cl::sycl::item<Rank> item) const
		{
			return {item, base::get_accessor()};
		}
	};

	template <typename T, int Rank, typename AccessorType, size_t... Extents>
	class accessor_proxy<T, Rank, AccessorType, chunk<T, Extents...>>
		: public accessor_proxy_base<AccessorType>
	{
	public:
		using base = accessor_proxy_base<AccessorType>;

		static_assert(sizeof...(Extents) == Rank, "must specify extent for every dimension");

		explicit accessor_proxy(AccessorType acc, cl::sycl::id<Rank>, cl::sycl::range<Rank>)
			: base(acc)
		{
		}

		chunk<T, Extents...> operator[](const cl::sycl::item<Rank> item) const
		{
			return {item, base::get_accessor()};
		}
	};

	template <typename ExecutionPolicy, cl::sycl::access::mode Mode, typename AccessorType, template <typename, int> typename Iterator, typename T, int Rank>
	auto create_accessor(celerity::handler &cgh, Iterator<T, Rank> beg, Iterator<T, Rank> end)
	{
		// TODO: move accessor creation into proxy
		if constexpr (traits::policy_traits<ExecutionPolicy>::is_distributed)
		{
			if constexpr (traits::get_accessor_type_<AccessorType>() != algorithm::detail::access_type::all)
			{
				return beg.get_buffer().template get_access<Mode>(cgh, traits::accessor_traits<Rank, AccessorType>::range_mapper());
			}
			else
			{
				if (is_subrange(beg, end))
				{
					return beg.get_buffer().template get_access<Mode>(cgh, celerity::access::fixed<Rank>({*beg, distance(beg, end)}));
				}
				else
				{
					return beg.get_buffer().template get_access<Mode>(cgh, traits::accessor_traits<Rank, AccessorType>::range_mapper());
				}
			}
		}
		else
		{
			static_assert(std::is_same_v<one_to_one, AccessorType> || traits::is_all_v<AccessorType>, "range mappers not supported for host access");
			return beg.get_buffer().template get_access<Mode>(cgh, distance(beg, end));
		}
	}

	template <typename ExecutionPolicy, cl::sycl::access::mode Mode, typename AccessorType, template <typename, int> typename Iterator, typename T, int Rank>
	auto get_access(celerity::handler &cgh, Iterator<T, Rank> beg, Iterator<T, Rank> end)
	{
		const auto acc = create_accessor<ExecutionPolicy, Mode, AccessorType>(cgh, beg, end);
		return accessor_proxy<T, Rank, decltype(acc), AccessorType>{acc, *beg, distance(beg, end)};
	}

} // namespace celerity::algorithm::detail

#endif // ACCESSOR_PROXY_H
