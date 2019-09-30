#ifndef POWER_ITERATION_HPP
#define POWER_ITERATION_HPP

#include "../../hw3/1-vectorNorms/vectorNorms.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
T power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v_k(A.size());

  random_double_fill(std::begin(v_k), std::end(v_k), -100, 100);

  for (auto i = 0u; i < MAX; ++i)
  {
    v_k = A * v_k;
    v_k = v_k / p_norm(v_k, 2);
  }

  auto pointwise = v_k * (A * v_k);
  auto lambda = std::accumulate(
    std::begin(pointwise), std::end(pointwise), T(0.0), [](auto acc, auto val) {
      return acc + val;
    });

  return lambda;
}

#endif
