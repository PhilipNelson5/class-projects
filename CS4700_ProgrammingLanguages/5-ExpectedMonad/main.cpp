#include "expected.hpp"
#include "type.hpp"
#include <exception>
#include <iostream>

int main(void)
{
  Expected<int> a = 1;
  Expected<int> b(2);
  std::cout << type_name<decltype(a+2)>() << std::endl;
  std::cout << a/b << std::endl;
  // std::cout << "a: " << a/b << "\nb: " << b << std::endl;
}
