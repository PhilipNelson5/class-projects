#ifndef PARALLLEL_LINEAR_SEARCH
#define PARALLLEL_LINEAR_SEARCH

#include "parallelLinearSearch.hpp"
#include "threadPool.hpp"
#include <atomic>
#include <vector>

template <typename T>
void linearSearch(std::vector<T> const &list, T search, int beg, int end, std::atomic<int> &found)
{
  for (int i = beg; i <= end && found == -1; ++i)
  {
    if (list[i] == search)
    {
      found = i;
      return;
    }
  }
}

template <typename T>
int parallelLinearSearch(std::vector<T> const &list, T search, int threads)
{
  std::atomic<int> found(-1);
  int beg = 0;
  int end = list.size();
  int step = (end - beg) / threads;
  {
    ThreadPool pool(threads);
    for (int i = beg; i < end; i += step)
    {
      if (i + step > end)
        pool.post([&list, search, i, end, &found]() { linearSearch(list, search, i, end, found); });
      else
        pool.post([&list, search, i, step, &found]() {
          linearSearch(list, search, i, i + step - 1, found);
        });
    }
    pool.start();
  }
  return (int)found;
}

#endif
