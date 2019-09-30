#include <iostream>
#include "maceps.hpp"

int main()
{
  auto [sprec, seps] = smaceps();
  auto [dprec, deps] = dmaceps();

  std::cout << "single\t" << sprec << '\t' << seps << '\n';
  std::cout << "double\t" << dprec << '\t' << deps << '\n';
}
