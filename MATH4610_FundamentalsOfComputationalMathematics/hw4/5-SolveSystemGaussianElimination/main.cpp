#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "solveLinearSysGaussianElim.hpp"
#include <iostream>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  Matrix<double> A = {{1, 2, 3, 4}, {4, 5, 6, 6}, {2, 5, 1, 2}, {7, 8, 9, 7}};

  std::vector<double> x = {4, 6, -7, 9};
  auto b = A * x;

  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual x\n" << x << std::endl;

  std::cout << " Calculated x\n"
            << solve_linear_system_gaussian_elimination(A, b) << std::endl;
}
