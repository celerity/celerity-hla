#ifndef CELERITY_HLA_CONCEPTS
#define CELERITY_HLA_CONCEPTS

#include <type_traits>

namespace celerity::hla
{
    template< class T, class U >
    concept same_as = std::is_same_v<T, U> &&std::is_same_v<U, T>;

    template <typename T, typename R = void, typename... Args>
    concept Callable = std::is_invocable_v<R(Args...)>;

    struct inactive_probe_t {};

    template<class T>
    concept InactiveProbe = same_as<std::decay_t<T>, inactive_probe_t>;    
    



}

#endif