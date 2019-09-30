#ifndef FIRST_ORDER_IVP_HPP
#define FIRST_ORDER_IVP_HPP

#include <cmath>
#include <functional>

template <typename T>
auto firstOrderIVPSolver(const T& l, const T& a)
{
  return [=](const T& t) { return a * std::exp(l * t); };
}

#endif
