#ifndef ELLIPTIC_ODE_HPP
#define ELLIPTIC_ODE_HPP

#include "finDiffCoeff.hpp"
#include <tuple>

template <std::size_t N, typename F, typename T>
std::
  tuple<std::array<T, N - 1>, std::array<T, N - 1>, std::array<T, N - 1>, std::array<T, N - 1>>
  initEllipticODE(F f, T a, T b, T ua, T ub)
{
  auto h = (b - a) / N;
  auto h2 = h * h;

  auto coeff = centralFinDiffCoeff<double, 2, 2>();

  std::array<T, N - 1> av, bv, cv, fv;

  av.fill(coeff[0]);
  bv.fill(coeff[1]);
  cv.fill(coeff[2]);

  for (auto i = 1u; i < N; ++i)
  {
    fv[i - 1] = h2 * f(a + i * h);
  }

  fv[0] -= ua;
  fv[fv.size() - 1] -= ub;

  return {av, bv, cv, fv};
}

template <std::size_t N, typename F, typename T>
std::array<T, N - 1> solveEllipticODE(F f, T a, T b, T ua, T ub)
{
  auto[_a, _b, _c, _f] = initEllipticODE<N>(f, a, b, ua, ub);

  return Matrix<double, N - 1, N - 1>::triDiagThomas(_a, _b, _c, _f);
}

  // template <std::size_t N, typename F, typename T>
  // std::array<T, N-1> solveEllipticK(F f, T a, T b, T ua, T ub, std::array<T, N-1> k)
  // {
  // auto h = (b - a) / N;
  // auto h2 = h * h;
  //
  // for (auto i = 1u; i < N; ++i)
  // {
  // fv[i - 1] = h2 * f(a + i * h);
  // }
  //
  // }

#endif
