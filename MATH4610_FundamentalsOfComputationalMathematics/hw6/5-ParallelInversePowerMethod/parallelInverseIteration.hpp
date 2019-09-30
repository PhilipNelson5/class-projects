#ifndef PARALLEL_INVERSE_POWER_ITERATION_HPP
#define PARALLEL_INVERSE_POWER_ITERATION_HPP

#include "../../hw3/1-vectorNorms/vectorNorms.hpp"
#include "../../hw4/7-SolveSystemLUFactorization/solveLinearSystemLU.hpp"
#include "../../hw3/7-parallelMatrixOperations/parallelMatrixOperations.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
T parallel_inverse_power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v(A.size());

  random_double_fill(std::begin(v), std::end(v), -100, 100);

  T lamda = 0.0;
  for (auto i = 0u; i < MAX; ++i)
  {
    auto w = solve_linear_system_LU(A, v);
    v = w / p_norm(w, 2);
    auto pointwise = v * parallel_multiply(A, v);
    lamda = std::accumulate(std::begin(pointwise),
                            std::end(pointwise),
                            T(0.0),
                            [](auto acc, auto val) { return acc + val; });
  }
  return lamda;
}

#endif
