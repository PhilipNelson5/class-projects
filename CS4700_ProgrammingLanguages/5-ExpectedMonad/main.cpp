#include "expected.hpp"
#include <iostream>

int main(void)
{
  Expected<int> a(2);
  std::cout << a << std::endl;
}
