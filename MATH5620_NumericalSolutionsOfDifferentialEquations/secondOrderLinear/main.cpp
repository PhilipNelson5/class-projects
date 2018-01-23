#include "solcc.hpp"
#include <iostream>

int main()
{
  auto solution = solcc(2.0, 0.0, 3.0, 5.0, -1.0, 3.0).value();
  std::cout << solution << std::endl;
}
