#ifndef GAUSS_SIDEL_HPP
#define GAUSS_SIDEL_HPP

#include "../machineEpsilon/maceps.hpp"
#include "../matrix/matrix.hpp"
#include "../matrix/vector_util.hpp"
#include <array>

template <typename T, std::size_t N>
std::array<T, N> gauss_sidel(Matrix<T, N, N>& A,
                             std::array<T, N> const& b,
                             unsigned int const& MAX_ITERATIONS = 1000u)
{
  auto ct = 0u;
  std::array<T, N> x, xn;
  x.fill(0);
  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    ++ct;
    xn.fill(0);
    for (auto i = 0u; i < N; ++i)
    {
      auto s1 = 0.0, s2 = 0.0;
      for (auto j = 0u; j < i; ++j)
      {
        s1 += A[i][j] * xn[j];
      }
      for (auto j = i + 1; j < N; ++j)
      {
        s2 += A[i][j] * x[j];
      }
      xn[i] = (b[i] - s1 - s2) / A[i][i];
    }
    if (allclose(x, xn, maceps<T>().maceps))
    {
      std::cout << "Gauss Sidel completed in " << ct << " iterations\n";
      return xn;
    }
    x = xn;
  }
  return x;
}

#endif
