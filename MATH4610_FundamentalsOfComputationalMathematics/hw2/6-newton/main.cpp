#include "newton.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto fprime = [](double x) { return 2 * x; };

  auto g = [](double x) { return sin(M_PI * x); };
  auto gprime = [](double x) { return M_PI * cos(M_PI * x); };

  auto root = root_finder_newton(f, fprime, 3.0, 1e-100, 100);
  std::cout << std::setprecision(15) << root << std::endl;

  root = root_finder_newton(g, gprime, 4.75, 1e-100, 100);
  std::cout << std::setprecision(15) << root << std::endl;
}
