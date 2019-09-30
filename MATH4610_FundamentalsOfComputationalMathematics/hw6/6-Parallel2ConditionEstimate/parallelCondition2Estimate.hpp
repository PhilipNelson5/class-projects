#ifndef PARALLEL_CONDITION_2_ESTIMATE_HPP
#define PARALLEL_CONDITION_2_ESTIMATE_HPP

#include "../4-ParallelPowerMethod/parallelPowerIteration.hpp"
#include "../5-ParallelInversePowerMethod/parallelInverseIteration.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
T parallel_condition_2_estimate(Matrix<T> const& A, unsigned int const& MAX)
{
  auto maxEig = parallel_power_iteration(A, MAX);
  auto minEig = parallel_inverse_power_iteration(A, MAX);
  return std::abs(maxEig / minEig);
}

#endif
