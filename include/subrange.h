#ifndef SUBRANGE_H
#define SUBRANGE_H

#include "packaged_task_traits.h"
#include "t_joint.h"

#include "linkage_traits.h"

namespace celerity::algorithm::detail {

template <typename... Actions>
auto resolve_subranges(const sequence<Actions...> &s);

namespace subrange_impl {
template <typename T, typename U> auto operator+(T lhs, U rhs) {
  return sequence(lhs, rhs);
}

template <typename U, typename... Ts>
auto operator+(const sequence<Ts...> &lhs, U rhs) {
  constexpr auto op = [](auto &&a, auto &&b) {
    return subrange_impl::operator+(std::forward<decltype(a)>(a),
                                    std::forward<decltype(b)>(b));
  };

  return apply_append(lhs, rhs, op);
}

template <typename... Actions, size_t... Is>
auto resolve_subranges(const sequence<Actions...> &s,
                       std::index_sequence<Is...>) {
  constexpr auto op = [](auto &&a, auto &&b) {
    return subrange_impl::operator+(std::forward<decltype(a)>(a),
                                    std::forward<decltype(b)>(b));
  };

  return left_fold(s, op);
}

} // namespace subrange_impl

template <typename... Actions>
auto resolve_subranges(const sequence<Actions...> &s) {
  using namespace subrange_impl;

  if constexpr (sizeof...(Actions) == 1) {
    if constexpr (!traits::is_t_joint_v<
                      traits::first_element_t<sequence<Actions...>>>) {
      return sequence(s);
    } else {
      return sequence(resolve_internally(get_first_element(s)));
    }
  } else {
    return resolve_subranges(s, std::make_index_sequence<sizeof...(Actions)>{});
  }
}

} // namespace celerity::algorithm::detail

#endif // !SUBRANGE_H