#ifndef KERNEL_H
#define KERNEL_H

#include "celerity.h"

template<typename View, typename F>
class kernel
{
public:
  using view_type = View;
  
  kernel(View view, F f)
    : view_(view), f_(f) {}
    
  void operator()(handler cgh) 
  {
    f_(cgh);
  }

private:
  View view_;
  F f_;
};

template<typename View, typename F>
constexpr auto make_kernel(View view, F f)
{
  return kernel<View, F>{view, f};
}

#endif