#include "error.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  double approx = 3.2;
  double value = 3.14159;

  std::cout << "Approximate: " << approx << "\nReal Value: " << value << std::endl
            << std::endl;
  std::cout << "Absolute: " << absoluteError(approx, value) << std::endl;
  std::cout << "Relative: " << relativeError(approx, value) << std::endl;
}
