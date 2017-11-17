#include "expected.hpp"
#include "type.hpp"
#include <typeinfo>
#include <iostream>
#include <exception>

int main(void)
{
  //Expected<int> a(std::domain_error("FOO"));
  Expected<double> a(2);
  Expected<int> b(3);
  std::cout << type_name<decltype(a-b)>() << std::endl;
  std::cout << a-b << std::endl;
  std::cout << type_name<decltype(b-a)>() << std::endl;
  std::cout << b-a << std::endl;
}
