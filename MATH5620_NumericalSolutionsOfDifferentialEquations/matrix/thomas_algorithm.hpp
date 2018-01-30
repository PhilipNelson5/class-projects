#ifndef THOMAS_ALGORITHM_HPP
#define THOMAS_ALGORITHM_HPP

#include "matrix.hpp"

template <typename T, std::size_t N>
std::Array<T, N> thomas_algorithm(Matrix<T, N, N> m, std::Array<T, N> d)
{
  std::Array<T, N> x;
  std::Array<T, N> a;
  std::Array<T, N> b;
  std::Array<T, N> c;

  for (auto i, j = 0u; i < N; ++i, ++j)
  {

    if (i == 0)
    {
      a[i] = 0;
      b[i] = m[i][j];
      c[i] = m[i][j + 1];
    }
    if (i == N - 1)
    {
      a[i] = m[i][j - 1];
      b[i] = m[i][j];
      c[i] = 0;
    }
    else
    {
      a[i] = m[i][j - 1];
      b[i] = m[i][j];
      c[i] = m[i][j + 1];
    }
  }

  for (auto k = 1u; k < N; ++k)
  {
    auto m = a[k] / b[k - 1];
    b[k] = b[k] - m * c[k - 1];
    d[k] = d[k] - m * d[k - 1];
  }
}

#endif
