#ifndef JACOBI_ITERATION_HPP
#define JACOBI_ITERATION_HPP

#include "../matrix/matrix.hpp"
#include "../matrix/vector_util.hpp"
#include <array>

template <typename T, std::size_t N>
std::array<T, N> jacobiIteration(Matrix<T, N, N> A,
                                 std::array<T, N> const& b,
                                 unsigned int const& MAX_ITERATIONS = 1000u)
{
  auto ct = 0u;
  std::array<T, N> zeros;
  zeros.fill(0);

  std::array<T, N> x = zeros;

  for (auto n = 0u; n < MAX_ITERATIONS; ++n)
  {
    ++ct;
    auto x_n = zeros;

    for (auto i = 0u; i < N; ++i)
    {
      T sum = 0;
      for (auto j = 0u; j < N; ++j)
      {
        if (j == i) continue;
        sum += A[i][j] * x[j];
      }
      x_n[i] = (b[i] - sum) / A[i][i];
    }

    if (allclose(x, x_n, maceps<T>().maceps))
    {
      std::cout << "Jacobi Iteration completed in " << ct << " iterations\n";
      return x_n;
    }

    x = x_n;
  }

  return x;
}

#endif
