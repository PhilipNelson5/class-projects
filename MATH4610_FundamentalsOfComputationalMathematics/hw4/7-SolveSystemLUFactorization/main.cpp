#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "solveLinearSystemLU.hpp"
#include <iostream>

int main()
{
  Matrix<double> A = {{1, 2, 3, 4}, {4, 5, 6, 6}, {2, 5, 1, 2}, {7, 8, 9, 7}};
  std::vector<double> x = {4, 7, 2, 5};
  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << " Calculated x\n";
  std::cout << solve_linear_system_LU(A, b) << std::endl;
}
