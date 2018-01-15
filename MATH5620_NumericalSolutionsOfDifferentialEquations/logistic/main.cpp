#include "logistic.hpp"
#include <iostream>

int main()
{
  double a = 1.0;
  double b = 2.0;
  double t = 50;
  double p0 = 10.0;

  std::cout << "alpha:\t" << a << "\nbeta:\t" << b << "\ntime:\t" << t << "\nP0:\t" << p0 << std::endl;

  std::cout << "------------\nresult:\t" << logistic(a, b, t, p0) << std::endl;
}
