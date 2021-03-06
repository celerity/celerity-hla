#ifndef ACCESSORS_H
#define ACCESSORS_H

#include <cmath>

#include "any_accessor.h"
#include "variant_item.h"

namespace celerity::algorithm
{

	namespace detail
	{

		struct sycl_marker
		{
			cl::sycl::id<0> _;
		};

		struct one_to_one
		{
		};

	} // namespace detail

	template <typename T, size_t Dim, bool Transpose = false>
	class slice
	{
	public:
		using value_type = T;

		template <int Rank, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
		slice(const cl::sycl::item<Rank> item, const cl::sycl::range<Rank> range, celerity::algorithm::detail::device_accessor<T, Rank, Mode, Target> acc)
			: idx_(static_cast<int>(item.get_id()[Dim])), range_(range[Dim]), item_(item), accessor_(acc)
		{
		}

		int index() const { return idx_; }
		int get_range() const { return range_; }

		T operator*() const
		{
			return this->operator[](idx_);
		}

		T operator[](int pos) const
		{
			return item_.apply([pos, this](const auto &item) {
				using id_type = std::decay_t<decltype(item.get_id())>;

				if constexpr (Transpose && std::is_same_v<id_type, cl::sycl::id<2>>)
				{
					auto id = item.get_id();
					id[1 - Dim] = id[Dim];
					id[Dim] = pos;

					return accessor_.template get(id);
				}
				else if constexpr (Transpose)
				{
					// TODO: unresolved external
					// assert(false && "Only slices of 2-dimensional ranges can be transposed");
					return T{};
				}
				else
				{
					auto id = item.get_id();
					id[Dim] = pos;
					return accessor_.template get(id);
				}
			});
		}

		slice(const slice<T, Dim> &) = delete;
		slice(slice<T, Dim> &&) = delete;
		slice<T, Dim> &operator=(const slice<T, Dim> &) = delete;
		slice<T, Dim> &operator=(slice<T, Dim> &&) = delete;

		template <typename V>
		slice<T, Dim> &operator=(const V &) = delete;

	private:
		const int idx_;
		const int range_;
		const variant_item<2, 3> item_;
		const detail::any_accessor<T> accessor_;
	};

	template <typename T, size_t Dim>
	using t_slice = slice<T, Dim, true>;

	template <typename T, size_t... Extents>
	class chunk
	{
	public:
		using value_type = T;

		static constexpr auto rank = sizeof...(Extents);

		template <typename AccessorType>
		chunk(cl::sycl::item<rank> item, AccessorType acc)
			: item_(item), accessor_(acc)
		{
		}

		auto item() const { return item_; }

		T operator*() const
		{
			return this->operator[]({});
		}

		T operator[](cl::sycl::rel_id<rank> rel_id) const
		{
			auto id = item_.get_id();
			
			for (auto i = 0u; i < rank; ++i)
			{
				id[i] = static_cast<size_t>(static_cast<long>(id[i]) + rel_id[i]);
			}

			return accessor_.template get(id);
		}

		chunk(const chunk<T, Extents...> &) = delete;
		chunk(chunk<T, Extents...> &&) = delete;
		chunk<T, Extents...> &operator=(const chunk<T, Extents...> &) = delete;
		chunk<T, Extents...> &operator=(chunk<T, Extents...> &&) = delete;

		template <typename V>
		chunk<T, Extents...> &operator=(const V &) = delete;

		T get(cl::sycl::id<rank> abs_id) const
		{
			return accessor_.template get(abs_id);
		}

		bool is_on_boundary() const
		{
			return dispatch_is_on_boundary(item_.get_range(), std::make_index_sequence<rank>());
		}

		bool is_on_boundary(cl::sycl::range<rank> range) const
		{
			return dispatch_is_on_boundary(range, std::make_index_sequence<rank>());
		}

		template <typename F, typename G,
				  std::enable_if_t<std::is_invocable_v<F> && std::is_invocable_v<G>, int> = 0>
		auto discern(F on_bounds_functor, G in_bounds_functor) const
		{
			return is_on_boundary()
					   ? std::invoke(on_bounds_functor)
					   : std::invoke(in_bounds_functor);
		}

		template <typename F, typename G,
				  std::enable_if_t<std::is_invocable_v<G> && std::is_same_v<F, std::invoke_result_t<G>>, int> = 0>
		auto discern(F on_bounds_value, G in_bounds_functor) const
		{
			return is_on_boundary()
					   ? on_bounds_value
					   : std::invoke(in_bounds_functor);
		}

		template <typename F, typename G,
				  std::enable_if_t<std::is_invocable_v<F> && std::is_same_v<G, std::invoke_result_t<F>>, int> = 0>
		auto discern(F on_bounds_functor, G in_bounds_value) const
		{
			return is_on_boundary()
					   ? std::invoke(on_bounds_functor)
					   : in_bounds_value;
		}

	private:
		const cl::sycl::item<rank> item_;
		const detail::any_accessor<T> accessor_;

		template <size_t... Is>
		bool dispatch_is_on_boundary(cl::sycl::range<rank> range, std::index_sequence<Is...>) const
		{
			const auto id = item_.get_id();

			return ((id[Is] < (Extents / 2)) || ...) ||
				   ((static_cast<int>(id[Is]) > static_cast<int>(range[Is]) - static_cast<int>(Extents / 2) - 1) || ...);
		}
	};

	template <typename T, int Rank>
	class all
	{
	public:
		using value_type = T;

		template <typename AccessorType>
		all(AccessorType acc, cl::sycl::id<Rank> offset, cl::sycl::range<Rank> range)
			: accessor_(acc), offset_(offset), range_(range)
		{
		}

		T operator[](cl::sycl::id<Rank> id) const
		{
			return accessor_.template get(id);
		}

		all(const all<T, Rank> &) = delete;
		all(all<T, Rank> &&) = delete;
		all<T, Rank> &operator=(const all<T, Rank> &) = delete;
		all<T, Rank> &operator=(all<T, Rank> &&) = delete;

		template <typename V>
		all<T, Rank> &operator=(const V &) = delete;

		auto get_range() const -> cl::sycl::range<Rank> { return range_; }
		auto get_offset() const -> cl::sycl::id<Rank> { return offset_; }

	private:
		const detail::sycl_marker _ = {};
		const detail::any_accessor<T> accessor_;
		const cl::sycl::id<Rank> offset_;
		const cl::sycl::range<Rank> range_;
	};

	namespace aliases
	{

		template <int Dim>
		using slice_f = slice<float, Dim>;

		template <int Dim>
		using slice_d = slice<double, Dim>;

		template <int Dim>
		using slice_i = slice<int, Dim>;

		template <int Dim>
		using slice_2f = slice<cl::sycl::float2, Dim>;

		template <int Dim>
		using slice_2d = slice<cl::sycl::double2, Dim>;

		template <int Dim>
		using slice_2i = slice<cl::sycl::int2, Dim>;

		template <int Dim>
		using slice_3f = slice<cl::sycl::float3, Dim>;

		template <int Dim>
		using slice_3d = slice<cl::sycl::double3, Dim>;

		template <int Dim>
		using slice_3i = slice<cl::sycl::int3, Dim>;

		template <size_t... Extents>
		using chunk_f = chunk<float, Extents...>;

		template <size_t... Extents>
		using chunk_d = chunk<double, Extents...>;

		template <size_t... Extents>
		using chunk_i = chunk<int, Extents...>;

		template <size_t... Extents>
		using chunk_2f = chunk<cl::sycl::float2, Extents...>;

		template <size_t... Extents>
		using chunk_2d = chunk<cl::sycl::double2, Extents...>;

		template <size_t... Extents>
		using chunk_2i = chunk<cl::sycl::int2, Extents...>;

		template <size_t... Extents>
		using chunk_3f = chunk<cl::sycl::float3, Extents...>;

		template <size_t... Extents>
		using chunk_3d = chunk<cl::sycl::double3, Extents...>;

		template <size_t... Extents>
		using chunk_3i = chunk<cl::sycl::int3, Extents...>;

		using all_f = all<float, 1>;

		using all_d = all<double, 1>;

		using all_i = all<int, 1>;

		using all_f2 = all<float, 2>;

		using all_d2 = all<double, 2>;

		using all_i2 = all<int, 2>;

		using all_f3 = all<float, 3>;

		using all_d3 = all<double, 3>;

		using all_i3 = all<int, 3>;

		using all_2f = all<cl::sycl::float2, 1>;

		using all_2d = all<cl::sycl::double2, 1>;

		using all_2i = all<cl::sycl::int2, 1>;

		using all_3f = all<cl::sycl::float3, 1>;

		using all_3d = all<cl::sycl::double3, 1>;

		using all_3i = all<cl::sycl::int3, 1>;

		using all_2f2 = all<cl::sycl::float2, 2>;

		using all_2d2 = all<cl::sycl::double2, 2>;

		using all_2i2 = all<cl::sycl::int2, 2>;

		using all_3f2 = all<cl::sycl::float3, 2>;

		using all_3d2 = all<cl::sycl::double3, 2>;

		using all_3i2 = all<cl::sycl::int3, 2>;

		using all_2f3 = all<cl::sycl::float2, 3>;

		using all_2d3 = all<cl::sycl::double2, 3>;

		using all_2i3 = all<cl::sycl::int2, 3>;

		using all_3f3 = all<cl::sycl::float3, 3>;

		using all_3d3 = all<cl::sycl::double3, 3>;

		using all_3i3 = all<cl::sycl::int3, 3>;

	} // namespace aliases

} // namespace celerity::algorithm

#endif // ACCESSORS_H