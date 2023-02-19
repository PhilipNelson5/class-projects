#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

using std::begin; 
using std::end;

template <typename T>
T euclidean_distance(const std::vector<T>& a, const std::vector<T>& b)
{
  assert(a.size() == b.size());

  std::vector<T> temp;
  temp.reserve(a.size());
  std::transform(begin(a), end(a), begin(b), std::back_inserter(temp), [](const T a, const T b){ return std::pow(a-b, 2); });
  return std::sqrt(std::accumulate(begin(temp), end(temp), T{0}));
}