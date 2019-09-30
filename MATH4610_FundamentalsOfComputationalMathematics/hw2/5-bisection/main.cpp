#include "bisection.hpp"
#include <iomanip>
#include <iostream>
#include <cmath>

int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto g = [](double x) { return sin(M_PI*x); };

  auto root = root_finder_bisection(f, 0.0, 5.5, 1e-100);
  if (root.has_value())
  {
    std::cout << std::setprecision(20) << root.value() << '\n';
  }
  else
  {
    std::cout << "no roots on specified interval\n";
  }

  root = root_finder_bisection(g, 4.5, 5.5, 1e-100);
  if (root.has_value())
  {
    std::cout << std::setprecision(20) << root.value() << '\n';
  }
  else
  {
    std::cout << "no roots on specified interval\n";
  }
}
