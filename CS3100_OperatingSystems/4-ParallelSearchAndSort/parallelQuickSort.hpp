#ifndef PARALLEL_QUICK_SORT_HPP
#define PARALLEL_QUICK_SORT_HPP

#include "threadPool.hpp"
#include <memory>
#include <vector>
#include <algorithm>

// choose the median of 3 for pivot
template <typename T>
int med(std::vector<T> &v, int start, int end)
{
  auto a = v[start];
  auto b = v[(start + end - 1) / 2];
  auto c = v[end];
  return max(min(a,b), min(max(a,b),c));
}

// divide the list and sort about the pivot
template <typename T>
int partition(std::vector<T> &v, int pivot, int start, int end)
{
  std::swap(v[pivot], v[end - 1]);
  int check = start;
  for (int i = start; i < end - 1; ++i)
  {
    if (v[i] <= v[end - 1])
    {
      std::swap(v[i], v[check]);
      ++check;
    }
  }
  std::swap(v[end - 1], v[check]);
  return check;
}

// recursive quick sort
template <typename T>
void quick(std::vector<T> &v, int start, int end, ThreadPool &pool)
{
  if (end - start <= 1) return;
  auto pivot = (start + end) / 2; // med(v, start, end) ;
  pivot = partition(v, pivot, start, end);
  pool.post([&v, start, pivot, &pool]() { quick(v, start, pivot, pool); });
  pool.post([&v, pivot, end, &pool]() { quick(v, pivot, end, pool); });
}

// call to recursive quick sort
template <typename T>
void parallelQuickSort(std::vector<T> &list, int threads)
{
  ThreadPool pool(threads);
  pool.post([&]() { quick(list, 0, list.size(), pool); });
  pool.start();
}

template <typename T>
bool isSorted(std::vector<T> &v)
{
  for (unsigned int i = 0; i < v.size() - 1; ++i)
    if (v[i] > v[i + 1]) return false;
  return true;
}
#endif
