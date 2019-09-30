#ifndef CONDITION_2_ESTIMATE_HPP
#define CONDITION_2_ESTIMATE_HPP

#include "../1-PowerMethod/powerIteration.hpp"
#include "../2-InversePowerMethod/inversePowerIteration.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
T condition_2_estimate(Matrix<T> const& A, unsigned int const & MAX)
{
  auto maxEig = power_iteration(A, MAX);
  auto minEig = inverse_power_iteration(A, MAX);
  return std::abs(maxEig / minEig);
}

#endif
