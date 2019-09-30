#include "heat.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  auto a = 0.7;
  auto F = 0.01;
  auto Nx = 10u;
  auto Tf = 1.0;
  auto L = 1.0;

  auto solution = heat_explicit_euler<double>(L, Nx, F, a, Tf);

  for(auto && x:solution)
    std::cout << x << '\n';

  return EXIT_SUCCESS;
}
