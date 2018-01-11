#include "maceps.hpp"
#include <iostream>
#include <limits>

int main()
{
  auto inteps = maceps<int>();
  std::cout << "int\n";
  std::cout << "precision:\t" << inteps.prec << std::endl;
  std::cout << "maceps:\t\t" << inteps.maceps << std::endl;
  std::cout << "std::numeric:\t" << std::numeric_limits<int>::epsilon() << std::endl << std::endl;

  auto floateps = maceps<float>();
  std::cout << "float\n";
  std::cout << "precision:\t" << floateps.prec << std::endl;
  std::cout << "maceps:\t\t" << floateps.maceps << std::endl;
  std::cout << "std::numeric:\t" << std::numeric_limits<float>::epsilon() << std::endl << std::endl;

  auto doubleeps = maceps<double>();
  std::cout << "double\n";
  std::cout << "precision:\t" << doubleeps.prec << std::endl;
  std::cout << "maceps:\t\t" << doubleeps.maceps << std::endl;
  std::cout << "std::numeric:\t" << std::numeric_limits<double>::epsilon() << std::endl << std::endl;

  auto longdoubleeps = maceps<long double>();
  std::cout << "long double\n";
  std::cout << "precision:\t" << longdoubleeps.prec << std::endl;
  std::cout << "maceps:\t\t" << longdoubleeps.maceps << std::endl;
  std::cout << "std::numeric:\t" << std::numeric_limits<long double>::epsilon() << std::endl << std::endl;

}
