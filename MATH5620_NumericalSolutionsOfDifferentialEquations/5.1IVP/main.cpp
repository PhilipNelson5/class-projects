#include <iostream>
#include <iomanip>
#include "firstOrderIVP.hpp"

int main()
{

  auto solveIVP = firstOrderIVPSolver(-1.5, 7.3);

  for(auto t = 0; t < 10; ++t)
  {
    std::cout << "t = " << t << " -> " << solveIVP(t)
      << "\tt = " << t+10 << " -> " << solveIVP(t+10) << '\n';
  }

  return EXIT_SUCCESS;
}
