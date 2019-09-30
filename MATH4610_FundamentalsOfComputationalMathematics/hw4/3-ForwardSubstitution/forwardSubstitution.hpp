#ifndef FORWARD_SUBSTITUTION_HPP
#define FORWARD_SUBSTITUTION_HPP

#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 *  Solves the system Ly = b
 *
 *  @tparam T The type of the elements in L and b
 *  @param L  A lower triangular matrix
 *  @param b  A vector
 *  @return A vector with the solution y
 */
template <typename T>
std::vector<T> forward_substitution(Matrix<T> L, std::vector<T> b)
{
  std::vector<T> y(b.size());

  for (auto i = 0u; i < L.size(); ++i)
  {
    T sum = 0.0;
    for (auto j = 0u; j < i; ++j)
    {
      sum += L[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }

  return y;
}

#endif
