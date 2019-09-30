//#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../utils/matrixUtils.hpp"
#include "choleskyFactorization.hpp"
#include <iostream>
#include <iomanip>

int main()
{
  const auto n = 5;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;
  auto L = cholesky_factorization(A);
  auto LLT = L * transpose(L);

  std::cout << " A\n" << A << std::endl;
  std::cout << " L\n" << L << std::endl;
  std::cout << " L*L^T\n" << LLT << std::endl;
}
