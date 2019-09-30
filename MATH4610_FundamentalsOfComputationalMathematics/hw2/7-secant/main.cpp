#include "secant.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto g = [](double x) { return sin(M_PI * x); };

  auto root = root_finder_secant(f, 4.0, 5.5, 1e-10);
  std::cout << std::setprecision(20) << root << '\n';

  root = root_finder_secant(g, 3.5, 6.5, 1e-10);
  std::cout << std::setprecision(20) << root << '\n';
}
