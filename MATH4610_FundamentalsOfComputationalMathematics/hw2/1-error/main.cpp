#include "error.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
  auto value = M_PI;
  auto approx = 3.1416;

  std::cout << std::setprecision(15) << "Real Value:  " << value
            << "\nApproximate: " << approx << std::endl
            << std::endl;

  std::cout << "Absolute Error: " << absolute_error(approx, value) << std::endl;
  std::cout << "Relative Error: " << relative_error(approx, value) << std::endl;
}
