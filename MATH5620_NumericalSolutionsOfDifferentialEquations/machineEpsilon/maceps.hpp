#ifndef MACEPS_HPP
#define MACEPS_HPP

#include <cmath>
template <typename T>
struct eps
{
  int prec;
  T maceps;
  eps(int& p, T e) : prec(p), maceps(e) {}
};

template <typename T>
eps<T> maceps()
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

  return eps(prec, e);
}

#endif
