#ifndef NEWTONS_METHOD
#define NEWTONS_METHOD

#include "../machineEpsilon/maceps.hpp"
#include <iomanip>
#include <iostream>

template <typename T, typename F>
T newtons_method(F f, T dt, T x0, const unsigned int MAX_ITERATONS = 100)
{
  auto tol = maceps<T>().maceps, i = 0u;
  while (std::abs(f(x0) - 0) > tol && ++i < MAX_ITERATONS)
  {
    x0 = x0 - f(x0) / ((f(x0 + dt) - f(x0)) / dt);
  }
  return x0;
}

#endif
