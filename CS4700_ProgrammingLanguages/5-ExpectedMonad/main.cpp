#include "expected.hpp"
#include "type.hpp"
#include <cmath>
#include <exception>
#include <iostream>

template <typename T>
Expected<double> root(T t)
{
  if (t < 0) return std::domain_error("Imaginary Root");
  return std::sqrt(t);
}

int main(void)
{
  Expected<int> a(1);
  Expected<int> b(2);
  Expected<float> c(2.2f);
  auto d = root(-1);
  std::cout << d + a << std::endl;
  std::cout << type_name<decltype(d)>() << std::endl << std::endl;
  std::cout << a << " + " << c << " = " << a + c << std::endl;
  std::cout << type_name<decltype(a + c)>() << std::endl << std::endl;
  std::cout << c << " + " << a << " = " << c + a << std::endl;
  std::cout << type_name<decltype(c + a)>() << std::endl << std::endl;
  std::cout << c << " + " << 1 << " = " << c + 1 << std::endl;
  std::cout << type_name<decltype(c + 1)>() << std::endl << std::endl;
  std::cout << 1 << " + " << c << " = " << 1 + c << std::endl;
  std::cout << type_name<decltype(1 + c)>() << std::endl << std::endl;
}
