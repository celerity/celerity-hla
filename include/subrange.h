#ifndef SUBRANGE_H
#define SUBRANGE_H

#include "packaged_task_traits.h"
#include "t_joint.h"

#include "linkage_traits.h"
#include "buffer_range.h"
#include "iterator_transform.h"

namespace celerity::hla::detail
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
              require<traits::is_celerity_buffer_v<T> || traits::is_buffer_range_v<T>,
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
                      !traits::is_buffer_range_v<T>,
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

    // TODO
    const auto seq = traverse(s, [](const auto &seq) { return resolve_subranges(seq); });
    // const auto seq = s;

    if constexpr (traits::size_v<decltype(seq)> == 1)
    {
      if constexpr (!traits::is_t_joint_v<
                        traits::first_element_t<decltype(seq)>>)
      {
        return sequence(seq);
      }
      else
      {
        return sequence(resolve_internally(get_first_element(seq)));
      }
    }
    else
    {
      return resolve_subranges(seq, std::make_index_sequence<traits::size_v<decltype(seq)>>{});
    }
  }

} // namespace celerity::hla::detail

#endif // !SUBRANGE_H