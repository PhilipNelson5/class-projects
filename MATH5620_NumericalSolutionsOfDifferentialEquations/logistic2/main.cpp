#include "logisticSolver.hpp"
#include <iostream>

int main()
{
  double a = 2.5;
  double b = 1.7;
  double p0 = 3.2;

  std::cout << "alpha:\t" << a << "\nbeta:\t" << b << "\nP0:\t" << p0 << "\n\n";

   auto solveLog = logisticSolver(a, b, p0);

  // Call it for some basic values
  for (int t = -10; t < 0; ++t) {
    std::cout << "t = " << t << " -> " << solveLog(t)
      << "\tt = " << t+10 << " -> " << solveLog(t+10) << '\n';
  }

  return EXIT_SUCCESS;
}
