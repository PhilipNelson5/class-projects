#ifndef EXPLICIT_EULER_HPP
#define EXPLICIT_EULER_HPP

#include "../machineEpsilon/maceps.hpp"
#include <iomanip>
#include <iostream>

template <typename T, typename F>
T explicit_euler(T x0, T y0, T x, T dt, F f)
{
  auto tol = maceps<T>().maceps;
  while (std::abs(x - x0) > tol)
  {
    y0 = y0 + (dt * f(x0, y0));
    x0 += dt;
  }
  return y0;
}

#endif
