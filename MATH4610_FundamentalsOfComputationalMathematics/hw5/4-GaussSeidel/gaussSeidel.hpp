#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

#include "../../hw1/1-maceps/maceps.hpp"
#include "../../utils/matrixUtils.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> gauss_seidel(Matrix<T>& A,
                           std::vector<T> const& b,
                           unsigned int const& MAX_ITERATIONS = 1000u)
{
  static const T macepsT = std::get<1>(maceps<T>());

  std::vector<T> x(b.size(), 0), x_new(b.size(), 0);

  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    std::fill(std::begin(x_new), std::end(x_new), 0);
    for (auto i = 0u; i < A.size(); ++i)
    {
      auto s1 = 0.0, s2 = 0.0;
      for (auto j = 0u; j < i; ++j)
      {
        s1 += A[i][j] * x_new[j];
      }
      for (auto j = i + 1; j < A.size(); ++j)
      {
        s2 += A[i][j] * x[j];
      }
      x_new[i] = (b[i] - s1 - s2) / A[i][i];
    }

    if (allclose(x, x_new, macepsT))
    {
      return x_new;
    }
    x = x_new;
  }
  return x;
}

#endif
