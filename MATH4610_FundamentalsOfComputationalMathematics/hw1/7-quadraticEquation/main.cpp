#include "quadraticEquation.hpp"
#include <iostream>

int main()
{
  auto res = quadratic_equation(2.0, 9.0, -5.0);
  if (res)
  {
    auto [r1, r2] = res.value();
    std::cout << "( " << r1 << " , " << r2 << " )\n";
  }
  else
  {
    std::cout << "imaginary roots\n";
  }

  res = quadratic_equation(2.0, 3.0, 5.0);
  if (res)
  {
    auto [r1, r2] = res.value();
    std::cout << r1 << ' ' << r2 << '\n';
  }
  else
  {
    std::cout << "imaginary roots\n";
  }
  return EXIT_SUCCESS;
}
