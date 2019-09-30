#include "fixedPointIteration.hpp"
#include <iostream>

int main()
{
  auto g1 = [](double x) { return x - (x * x - 3) / 10; };
  auto g2 = [](double x) { return x - sin(M_PI * x) / 2; };

  auto approx1 = root_finder_fixed_point_iteration(g1, 5.3, 1.0e-5);
  std::cout << approx1 << '\n';

  auto approx2 = root_finder_fixed_point_iteration(g2, 5.8, 1.0e-5);
  std::cout << approx2 << '\n';
}
