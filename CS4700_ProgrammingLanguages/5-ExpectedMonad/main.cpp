#include "expected.hpp"
#include "type.hpp"
#include <exception>
#include <iostream>
#include <cmath>

template <typename T>
Expected<double> root(T t)
{
  if (t<0) return std::domain_error("Imaginary Root");
  return std::sqrt(t);
}

int main(void)
{
  Expected<int> a(1);
  Expected<int> b(2);
  Expected<float> c(2.2f);
  auto d = root(-1);
  std::cout << type_name<decltype(d)>() << std::endl;
  std::cout << d+a << std::endl;
  std::cout << type_name<decltype(c+a)>() << std::endl;
  std::cout << c+a << std::endl;
  //std::cout << type_name<decltype(2 + a)>() << std::endl;
  //std::cout << type_name<decltype(a+2)>() << std::endl;
  //std::cout << type_name<decltype(a+b)>() << std::endl;
  //std::cout << a << " + " << c << " = " << a+c  << " : " << type_name<decltype(a+c)>() << std::endl;
  //std::cout << type_name<decltype(c+a)>() << std::endl;
  // std::cout << a+1 << '\n' << 1+a << '\n' << a+b << '\n' << b+a << std::endl;
  // std::cout << a/b << std::endl;
  // std::cout << "a: " << a/b << "\nb: " << b << std::endl;
}
