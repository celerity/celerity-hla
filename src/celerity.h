#ifndef CELERITY_H
#define CELERITY_H

struct handler{ int invocations; };

class distr_queue
{ 
public:
  template<typename F>
  void submit(F f)
  {
    f(handler{++invocation_count_});
  }

private:
  int invocation_count_ = 0;
};

#endif