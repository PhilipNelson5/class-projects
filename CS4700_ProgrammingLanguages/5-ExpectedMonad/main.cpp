#include "expected.hpp"
#include <iostream>
#include <exception>

int main(void)
{
  Expected<int> a(2);
  Expected<int> b(3);
  Expected<int> c = 1 + a;
  //std::cout <<  << std::endl;
}
