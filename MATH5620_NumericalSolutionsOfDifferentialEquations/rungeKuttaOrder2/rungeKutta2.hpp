#ifndef RUNGE_KUTTA_2_HPP
#define RUNGE_KUTTA_2_HPP

#include "../machineEpsilon/maceps.hpp"
#include <iomanip>
#include <iostream>

template <typename T, typename F>
T runge_kutta_order2(F f, unsigned int n, T h, T x0, T y0)
{
  long double y[n], x[n], k[n][n];

  x[0] = x0;
  y[0] = y0;

  for (auto i = 1u; i <= n; i++)
  {
    x[i] = x[i - 1] + h;
  }

  for (auto j = 1u; j <= n; j++)
  {
    k[1][j] = h * f(x[j - 1], y[j - 1]);
    std::cout << "K[1] = " << k[1][j] << "\t";
    k[2][j] = h * f(x[j - 1] + h, y[j - 1] + k[1][j]);
    std::cout << "K[2] = " << k[2][j] << "\n";
    y[j] = y[j - 1] + ((k[1][j] + k[2][j]) / 2);
    std::cout << "y[" << h * j << "] = " << std::setprecision(5) << y[j]
              << std::endl;
  }
  return x;
}

#endif
