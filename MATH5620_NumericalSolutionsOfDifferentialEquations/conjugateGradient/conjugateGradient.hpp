#ifndef CONJUGATE_GRADIENT_HPP
#define CONJUGATE_GRADIENT_HPP

#include "../machineEpsilon/maceps.hpp"
#include "../matrix/matrix.hpp"
#include "../matrix/vector_util.hpp"
#include <array>

template <typename T, std::size_t N>
std::array<T, N> conjugate_gradient(Matrix<T, N, N>& A,
                                    std::array<T, N> const& b,
                                    unsigned int const& MAX_ITERATIONS = 1000u)
{
  auto ct = 0u;
  auto tol = maceps<T>().maceps;
  std::array<T, N> x_k, x_k1;
  x_k.fill(0);
  x_k1.fill(0);
  auto r_k = b;
  auto r_k1 = r_k, r_k2 = r_k;
  auto p_k = r_k, p_k1 = r_k;
  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    ++ct;
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
      std::cout << "Conjugate Gradient completed in " << ct << " iterations\n";
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
