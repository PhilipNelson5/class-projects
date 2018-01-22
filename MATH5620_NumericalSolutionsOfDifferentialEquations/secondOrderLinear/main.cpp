#include "solcc.hpp"
#include <iostream>

int main()
{
  auto result = solcc(1.0, 0.0, 2.0, 3.0, -2.0, 4.0).value();
  std::cout << result << std::endl;
}
