#ifndef UPWINDING_HPP
#define UPWINDING_HPP

#include "../matrix/matrix.hpp"
#include "../matrix/matrix_util.hpp"
#include <iomanip>
#include <iostream>

template <std::size_t S, typename T, typename F>
std::vector<std::array<T, S - 2>> upwinding(const T xDomain[],
                                            const T tDomain[],
                                            const T dx,
                                            const T dt,
                                            F eta,
                                            const T c)
{
  std::array<T, S - 2> U;
  for (auto i = 0u; i < S - 2; ++i)
  {
    auto A = xDomain[0];
    auto x = A + (i + 1) * dx;
    U[i] = eta(x);
  }

  std::vector<double> coeffs = {c * dx / dt, 1 - c * dx / dt};

  auto A = makeNDiag<double, S - 2, S - 2>(coeffs, -1u);

  std::vector<std::array<T, S - 2>> solution = {U};

  for (auto t = tDomain[0]; t < tDomain[1]; t += dt)
  {
    U = A * U;
    solution.push_back(U);
  }

  return solution;
}

#endif
