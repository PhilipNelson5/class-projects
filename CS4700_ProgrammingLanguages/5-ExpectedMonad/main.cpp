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

  std::cout << std::boolalpha << "Expected<int> a = " << a << std::endl
            << "Expected<int> b = " << b << std::endl
            << "Expected<double> c = " << c << std::endl
            << "Expected<bool> d = " << d << std::endl
            << std::endl;

  std::cout << "ADDITION" << std::endl;
  std::cout << "a + b = " << a - b << std::endl << std::endl;

  std::cout << "SUBTRACTION" << std::endl;
  std::cout << "c - a = " << c - a << std::endl << std::endl;

  std::cout << "MULTIPLICATION" << std::endl;
  std::cout << "c * b = " << c * b << std::endl << std::endl;

  std::cout << "DIVISIONS" << std::endl;
  std::cout << "c / c = " << c / c << std::endl << std::endl;

  std::cout << "MODULO" << std::endl;
  std::cout << "a % b = " << a % b << std::endl << std::endl;

  std::cout << "GREATER" << std::endl;
  std::cout << "a > b = " << (a > b) << std::endl << std::endl;

  std::cout << "GREATER THAN EQUAL" << std::endl;
  std::cout << "a >= b = " << (a >= b) << std::endl << std::endl;

  std::cout << "LESS THAN" << std::endl;
  std::cout << "a < b = " << (a < b) << std::endl << std::endl;

  std::cout << "LESS THAN EQUAL" << std::endl;
  std::cout << "a <= b = " << (a <= b) << std::endl << std::endl;

  std::cout << "EQUALITY" << std::endl;
  std::cout << "a == b = " << (a == b) << std::endl << std::endl;

  std::cout << "ERROR" << std::endl;
  auto e = root(a);
  std::cout << "square root of a = " << e << std::endl << std::endl;
}
