#include "hybridSecant.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  auto f = [](double x) { return x * x - 3; };

  auto g = [](double x) { return sin(M_PI * x); };

  auto root = root_finder_hybrid_secant(f, 1.0, 10.0, 1e-10);
  if (root.has_value())
    std::cout << std::setprecision(15) << root.value() << std::endl;
  else
    std::cout << "error\n";

  root = root_finder_hybrid_secant(g, 4.1, 5.9, 1e-10);
  if (root.has_value())
    std::cout << std::setprecision(15) << root.value() << std::endl;
  else
    std::cout << "error\n";
}
