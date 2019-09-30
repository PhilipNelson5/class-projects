#include "maceps.hpp"
#include <iostream>
#include <limits>

int main()
{
  auto [float_prec, float_eps] = maceps<float>();
  std::cout << "float\n";
  std::cout << "precision:\t" << float_prec << '\n';
  std::cout << "maceps:\t\t" << float_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<float>::epsilon()
            << "\n\n";

  auto [double_prec, double_eps] = maceps<double>();
  std::cout << "double\n";
  std::cout << "precision:\t" << double_prec << '\n';
  std::cout << "maceps:\t\t" << double_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<double>::epsilon()
            << "\n\n";

  auto [long_double_prec, long_double_eps] = maceps<long double>();
  std::cout << "long double\n";
  std::cout << "precision:\t" << long_double_prec << '\n';
  std::cout << "maceps:\t\t" << long_double_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<long double>::epsilon()
            << "\n\n";

  return EXIT_SUCCESS;
}
