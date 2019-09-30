#include "../matrix/matrix.hpp"
#include "../matrix/matrix_util.hpp"
#include "../matrix/vector_util.hpp"
#include "ellipticODE.hpp"
#include "finDiffCoeff.hpp"
#include <cmath>
#include <iostream>

double exact(double x)
{
  return 2.5 + 2.5 * x - 1 / (M_PI * M_PI) * std::sin(M_PI * x);
}

int main()
{
  constexpr int N = 10;
  auto a = 0.0, b = 1.0, ua = 2.5, ub = 5.0;
  auto h = (b - a) / N;

  // auto[_a, _b, _c, _f] = initEllipticODE<N>([](double x) { return std::sin(M_PI * x); }, a,
  // b, ua, ub);
  //
  // std::cout << "a\n" << _a << std::endl;
  // std::cout << "b\n" << _b << std::endl;
  // std::cout << "c\n" << _c << std::endl;
  // std::cout << "f\n" << _f << std::endl;

  auto calcVal =
    solveEllipticODE<N>([](double x) { return std::sin(M_PI * x); }, a, b, ua, ub);

  std::cout << calcVal << std::endl;

  std::array<double, N - 1> realVal;
  for (auto i = 1u; i < N; ++i)
  {
    realVal[i - 1] = exact(a + i * h);
  }

  std::cout << realVal << std::endl;

  std::cout << "P1 norm:  " << pNorm(realVal - calcVal, 1) << std::endl;
  std::cout << "P2 norm:  " << pNorm(realVal - calcVal, 2) << std::endl;
  std::cout << "inf norm: " << infNorm(realVal - calcVal) << std::endl;

  return EXIT_SUCCESS;
}
