#include "explicitEuler.hpp"
#include <iostream>

int main() {
  std::cout << explicit_euler(0.0, -1.0, 0.4, 0.1, [](double a, double b){return a*a+2*b;}) << std::endl;
}
