#ifndef DECORATED_TASK_H
#define DECORATED_TASK_H

#include "iterator.h"
#include "celerity_helper.h"

namespace celerity::algorithm
{

template <typename T, int Rank>
class decorated_task
{
public:
    using task_type = T;

    decorated_task(task_type task, iterator<Rank> beg, iterator<Rank> end)
        : task_(task), beg_(beg), end_(end)
    {
    }

    void operator()(celerity::distr_queue &queue)
    {
        std::invoke(task_, queue, beg_, end_);
    }

private:
    task_type task_;
    iterator<Rank> beg_;
    iterator<Rank> end_;
};

template <typename T, int Rank>
decorated_task(T task, iterator<Rank> beg, iterator<Rank> end)->decorated_task<T, Rank>;

} // namespace celerity::algorithm

#endif // DECORATED_TASK_H