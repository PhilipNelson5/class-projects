#include "hybridNewton.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto fprime = [](double x) { return 2 * x; };

  auto g = [](double x) { return sin(M_PI * x); };
  auto gprime = [](double x) { return M_PI * cos(M_PI * x); };

  auto root = root_finder_hybrid_newton(f, fprime, 1.0, 10.0, 1e-100);
  std::cout << std::setprecision(15) << root << std::endl;

  root = root_finder_hybrid_newton(g, gprime, 4.1, 5.9, 1e-100);
  std::cout << std::setprecision(15) << root << std::endl;
}
