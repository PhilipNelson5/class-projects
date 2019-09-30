#include "error.hpp"
#include <iostream>
#include <cmath>

int main()
{
  auto approx = 3.1416;
  auto value = M_PI;

  std::cout << "Approximate: " << approx << "\nReal Value: " << value
            << std::endl
            << std::endl;
  std::cout << "Absolute: " << absoluteError(approx, value) << std::endl;
  std::cout << "Relative: " << relativeError(approx, value) << std::endl;
}
