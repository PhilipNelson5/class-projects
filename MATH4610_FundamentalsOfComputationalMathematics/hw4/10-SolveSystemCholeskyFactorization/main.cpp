#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../utils/matrixUtils.hpp"
#include "solveLinearSystemCholesky.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  const auto n = 5;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;

  std::vector<double> x = {4, 7, 2, 5, 4};
  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << " Calculated x\n";
  std::cout << solve_linear_system_cholesky(A, b) << std::endl;
}
