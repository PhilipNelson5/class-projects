#include <tuple>
#include "maceps.hpp"

std::tuple<int, float> smaceps()
{
  float e = 1;
  float one = 1;
  float half = 0.5;
  int prec = 1;
  while (one + e * half > one)
  {
    e *= half;
    ++prec;
  }

  return std::make_tuple(prec, e);
}

std::tuple<int, double> dmaceps()
{
  double e = 1;
  double one = 1;
  double half = 0.5;
  int prec = 1;
  while (one + e * half > one)
  {
    e *= half;
    ++prec;
  }

  return std::make_tuple(prec, e);
}
