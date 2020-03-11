#ifndef POLICY_TRAITS_H
#define POLICY_TRAITS_H

namespace celerity::algorithm
{

template <typename Policy>
struct policy_traits;

template <typename T>
struct decay_policy
{
    using type = std::decay_t<T>;
};

template <typename T>
using decay_policy_t = typename decay_policy<T>::type;

template <typename T>
struct strip_queue
{
    using type = T;
};

template <typename T>
using strip_queue_t = typename strip_queue<T>::type;

} // namespace celerity::algorithm

#endif // POLICY_TRAITS_H