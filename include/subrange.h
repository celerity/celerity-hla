#ifndef SUBRANGE_H
#define SUBRANGE_H

#include "packaged_task_traits.h"
#include "t_joint.h"

#include "linkage_traits.h"
#include "buffer_range.h"
#include "iterator_transform.h"

namespace celerity::algorithm::detail
{
  template <typename... Actions>
  auto resolve_subranges(const sequence<Actions...> &s);

  namespace subrange_impl
  {
    template <typename T, typename U,
              require<!traits::is_iterator_transform_v<T>,
                      !traits::is_iterator_transform_v<U>> = yes>
    auto operator+(T lhs, U rhs)
    {
      return sequence(lhs, rhs);
    }

    template <typename T, typename U,
              require<traits::is_celerity_buffer_v<T>,
                      traits::is_iterator_transform_v<U>> = yes>
    auto operator+(T lhs, U rhs)
    {
      auto it_beg = begin(lhs);
      auto it_end = end(lhs);

      std::invoke(rhs, it_beg, it_end);

      return sequence(buffer_range{it_beg, it_end});
    }

    template <int Rank>
    auto operator+(iterator_transform<Rank> lhs, iterator_transform<Rank> rhs)
    {
      return iterator_transform<Rank>{
          [=](auto &beg, auto &end) {
            std::invoke(lhs, beg, end);
            std::invoke(rhs, beg, end);
          }};
    }

    template <typename T, typename U,
              require<!traits::is_celerity_buffer_v<T>,
                      traits::is_iterator_transform_v<U>> = yes>
    auto operator+(T lhs, U rhs)
    {
      static_assert(std::is_void_v<T>, "subrange specifiers may only occur after buffers");
    }

    template <typename T, typename U,
              require<traits::is_iterator_transform_v<T>,
                      !traits::is_iterator_transform_v<U>> = yes>
    auto operator+(T lhs, U rhs)
    {
      static_assert(std::is_void_v<T>, "subrange specifiers may only occur after buffers");
    }

    template <typename U, typename... Ts>
    auto operator+(const sequence<Ts...> &lhs, U rhs)
    {
      constexpr auto op = [](auto &&a, auto &&b) {
        return subrange_impl::operator+(std::forward<decltype(a)>(a),
                                        std::forward<decltype(b)>(b));
      };

      return apply_append(lhs, rhs, op);
    }

    template <typename... Actions, size_t... Is>
    auto resolve_subranges(const sequence<Actions...> &s,
                           std::index_sequence<Is...>)
    {
      constexpr auto op = [](auto &&a, auto &&b) {
        return subrange_impl::operator+(std::forward<decltype(a)>(a),
                                        std::forward<decltype(b)>(b));
      };

      return left_fold(s, op);
    }

  } // namespace subrange_impl

  template <typename... Actions>
  auto resolve_subranges(const sequence<Actions...> &s)
  {
    using namespace subrange_impl;

    if constexpr (sizeof...(Actions) == 1)
    {
      if constexpr (!traits::is_t_joint_v<
                        traits::first_element_t<sequence<Actions...>>>)
      {
        return sequence(s);
      }
      else
      {
        return sequence(resolve_internally(get_first_element(s)));
      }
    }
    else
    {
      return resolve_subranges(s, std::make_index_sequence<sizeof...(Actions)>{});
    }
  }

} // namespace celerity::algorithm::detail

namespace celerity::algorithm
{
  template <size_t Rank>
  auto skip(cl::sycl::id<Rank> distance)
  {
    using namespace detail;

    return iterator_transform<Rank>{
        [=](auto &beg, auto) { beg += distance; }};
  }

  template <size_t Rank>
  auto take(cl::sycl::id<Rank> distance)
  {
    using namespace detail;

    return iterator_transform<Rank>{
        [=](auto &beg, auto &end) {
          auto tmp = beg;
          tmp += distance;
          end = tmp;
        }};
  }
} // namespace celerity::algorithm

#endif // !SUBRANGE_H