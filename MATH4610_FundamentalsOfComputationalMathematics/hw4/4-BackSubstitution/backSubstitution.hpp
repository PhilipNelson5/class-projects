#ifndef BACK_SUBSTITUTION_HPP
#define BACK_SUBSTITUTION_HPP

#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 *  Solves the system Ux = b
 *
 *  @tparam T The type of the elements in U and b
 *  @param U  A lower triangular matrix
 *  @param b  A vector
 *  @return A vector with the solution x
 */
template <typename T>
std::vector<T> back_substitution(Matrix<T> U, std::vector<T> b)
{
  std::vector<T> x(b.size());

  for (auto i = (int)U[0].size() - 1; i >= 0; --i)
  {
    T sum = 0.0;
    for (auto j = (unsigned)i + 1; j < U[0].size(); ++j)
    {
      sum += U[i][j] * x[j];
    }
    x[i] = (b[i] - sum) / U[i][i];
  }

  return x;
}

#endif
