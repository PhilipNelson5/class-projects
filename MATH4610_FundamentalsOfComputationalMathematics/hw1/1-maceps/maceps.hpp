#ifndef MACEPS_HPP
#define MACEPS_HPP

#include <tuple>

template <typename T>
constexpr std::tuple<int, T> maceps()
{
  T e = 1;
  T one = 1;
  T half = 0.5;
  int prec = 1;
  while (one + e * half > one)
  {
    e *= half;
    ++prec;
  }

  return std::make_tuple(prec, e);
}

#endif
