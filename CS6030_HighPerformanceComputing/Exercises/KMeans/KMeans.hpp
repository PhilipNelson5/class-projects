#pragma once

#include <vector>
#include "../../lib/random.hpp"

template <typename T>
std::vector<T> extent(std::vector<T> data)
{
  std::vector<T> extents;
  const int dimensions = data[0].size();
}

template <typename T>
void k_means(const std::vector<std::vector<T>>& data, const unsigned k)
{
  const int dimensions = data[0].size();
  const std::vector<T> extents = extent(data);
  // pick centroids
  std::vector<std::vector<T>> centroids;
  for (unsigned i = 0; i < k; ++i)
  {
    // centroids.push_back()
  }
  
  // cluster 

  // adjust centroids
}