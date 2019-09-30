#ifndef STEEPEST_DESCENT_HPP
#define STEEPEST_DESCENT_HPP

#include "../../hw1/1-maceps/maceps.hpp"
#include "../../hw3/1-vectorNorms/vectorNorms.hpp"
#include "../../utils/matrixUtils.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> steepest_descent(Matrix<T>& A,
                                std::vector<T> const& b,
                                unsigned int const& MAX_ITERATIONS = 1000u)
{
  static const T tol = std::get<1>(maceps<T>());
  std::vector<T> x_k(A.size(), 0), x_k1(A.size(), 0);
  std::vector<T> r_k = b;
  std::vector<T> r_k1 = r_k, r_k2 = r_k;
  std::vector<T> p_k = r_k, p_k1 = r_k;

  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    if (k != 0)
    {
      auto b_k = (r_k1 * r_k1) / (r_k2 * r_k2);
      p_k = r_k1 + b_k * p_k1;
    }

    auto s_k = A * p_k;
    auto a_k = r_k1 * r_k1 / (p_k * s_k);
    x_k = x_k1 + a_k * p_k;
    r_k = r_k1 - a_k * s_k;

    if (allclose(x_k, x_k1, tol))
    {
      return x_k;
    }

    r_k2 = r_k1;
    r_k1 = r_k;
    x_k1 = x_k;
    p_k1 = p_k;
  }

  return x_k;
}

#endif
