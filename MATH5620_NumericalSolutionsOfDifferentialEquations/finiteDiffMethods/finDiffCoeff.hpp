#ifndef FIN_DIFF_COEFF_HPP
#define FIN_DIFF_COEFF_HPP

#include "../matrix/matrix.hpp"
#include <cmath>
#include <vector>

unsigned int fact(unsigned int const& n)
{
  if (n == 0)
    return 1;
  unsigned int f = 1u;
  for (auto i = 1u; i <= n; ++i)
    f *= i;
  return f;
}

unsigned int binCoeff(unsigned int const& n, unsigned int const& k)
{
  auto nf = fact(n);
  auto kf = fact(k);
  auto nkf = fact(n - k);

  return nf / (kf * nkf);
}

template <typename T, std::size_t ord, std::size_t acc>
auto centralFinDiffCoeff()
{
  constexpr int size = 2.0 * std::floor((ord + 1.0) / 2.0) - 1.0 + acc;
  constexpr int P = (size - 1.0) / 2.0;

  Matrix<double, size, size> mat;
  for (auto i = 0; i < size; ++i)
  {
    for (auto j = 0; j < size; ++j)
    {
      mat[i][j] = std::pow(-P + j, i);
    }
  }

  std::array<T, size> b;
  b.fill(0.0);
  b[ord] = fact(ord);

  return mat.solveLinearSystemLU(b);
}

template <typename T, std::size_t N>
Matrix<T, N, N> SecOrdFinDifMethEllipticMat()
{
  auto coeff = centralFinDiffCoeff<double, 2, 2>();
  return Matrix<T, N, N> ([&](int i, int j){
      if (i == j) return coeff[1];
      if (i+1 == j) return coeff[0];
      if (i == j+1) return coeff[2];
      return 0.0;
    });
}
#endif
