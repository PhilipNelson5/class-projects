#ifndef LOGISTIC_SOLVER_HPP
#define LOGISTIC_SOLVER_HPP

#include <cmath>

template <typename T>
auto logisticSolver(T const& a, T const& b, T const& p0)
{
  return [=](T const& t) {
    return (a / (((a - p0 * b) / p0) * std::exp(-a * t) + b));
  };
}

#endif
