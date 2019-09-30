#include "../5-matrixOperations/matrixOperations.hpp"
#include "matrixNorms.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  std::vector<std::vector<int>> m = {{-3, 5, 7}, {2, 6, 4}, {0, 2, 8}};
  std::cout << m << std::endl;
  std::cout << "one norm: " << one_norm(m) << std::endl;
  std::cout << "inf norm: " << inf_norm(m) << std::endl;
  std::cout << "Frobenius norm: " << frobenius_norm(m) << std::endl;
}
