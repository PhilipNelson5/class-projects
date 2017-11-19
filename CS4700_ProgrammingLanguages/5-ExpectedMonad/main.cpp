#include "expected.hpp"
#include "type.hpp"
#include <cmath>
#include <exception>
#include <iostream>

template <typename X>
Expected<double> root(X x)
{
  if (x < 0) return std::domain_error("IMAGINARY ROOT");
  return sqrt(x);
}

int main(void)
{
  Expected<int> a(-1);
  Expected<int> b(2);
  Expected<double> c(4.56);
  Expected<bool> d(false);

  // clang-format off
  std::cout << std::boolalpha 
            << "Expected<int> a = " << a << std::endl
            << "Expected<int> b = " << b << std::endl
            << "Expected<double> c = " << c << std::endl
            << "Expected<bool> d = " << d << std::endl
            << std::endl;
  // clang-format on

  std::cout << "ADDITION" << std::endl;
  std::cout << "c + b = " << c + b << std::endl << std::endl;
  std::cout << "b + c = " << b + c << std::endl;

  std::cout << "SUBTRACTION" << std::endl;
  std::cout << "c - a = " << c - a << std::endl;
  std::cout << "c - 2 = " << c - 2 << std::endl << std::endl;

  std::cout << "MULTIPLICATION" << std::endl;
  std::cout << "c * b = " << c * b << std::endl;
  std::cout << "3 * b = " << 3 * b << std::endl << std::endl;

  std::cout << "DIVISION" << std::endl;
  std::cout << "c / c = " << c / c << std::endl;
  std::cout << "4 / c = " << 4 / c << std::endl << std::endl;

  std::cout << "MODULO" << std::endl;
  std::cout << "a % b = " << a % b << std::endl;
  std::cout << "3 % b = " << 3 % b << std::endl << std::endl;

  std::cout << "GREATER" << std::endl;
  std::cout << "a > b = " << (a > b) << std::endl;
  std::cout << "a > b = " << (a > 7) << std::endl << std::endl;

  std::cout << "GREATER THAN EQUAL" << std::endl;
  std::cout << "a >= b = " << (a >= b) << std::endl;
  std::cout << "4 >= b = " << (4 >= b) << std::endl << std::endl;

  std::cout << "LESS THAN" << std::endl;
  std::cout << "a < b = " << (a < b) << std::endl;
  std::cout << "7 < b = " << (7 < b) << std::endl << std::endl;

  std::cout << "LESS THAN EQUAL" << std::endl;
  std::cout << "a <= b = " << (a <= b) << std::endl;
  std::cout << "a <= 8.0 = " << (a <= 8.0) << std::endl << std::endl;

  std::cout << "EQUALITY" << std::endl;
  std::cout << "a == b = " << (a == b) << std::endl;
  std::cout << "a == 7.3f = " << (a == 7.3f) << std::endl << std::endl;

  std::cout << "ERROR" << std::endl;
  std::cout << "square root of a = " << root(a) << std::endl;
  std::cout << "square root of -5 = " << root(-5) << std::endl << std::endl;
  std::cout << "square root of 5 = " << root(5) << std::endl << std::endl;
}
