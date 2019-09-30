#ifndef JACOBI_ITERATION_HPP
#define JACOBI_ITERATION_HPP

#include "../../hw1/1-maceps/maceps.hpp"
#include "../../utils/matrixUtils.hpp"
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> jacobi_iteration(Matrix<T> A,
                                std::vector<T> const& b,
                                unsigned int const& MAX_ITERATIONS = 1000u)
{
  std::vector<T> x_new(b.size(), 0);
  std::vector<T> x(b.size(), 0);
  static const T macepsT = std::get<1>(maceps<T>());

  for (auto n = 0u; n < MAX_ITERATIONS; ++n)
  {
    std::fill(std::begin(x_new), std::end(x_new), 0);

    for (auto i = 0u; i < A.size(); ++i)
    {
      T sum = 0.0;
      for (auto j = 0u; j < A.size(); ++j)
      {
        if (j == i) continue;
        sum += A[i][j] * x[j];
      }
      x_new[i] = (b[i] - sum) / A[i][i];
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
