#ifndef CELERITY_H
#define CELERITY_H

struct handler{};

class distr_queue
{ 
public:
  template<typename F>
  void submit(F f)
  {
    f(handler{});
  }
};

#endif