#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "LUFactorization.hpp"
#include <iostream>

int main()
{
  Matrix<double> A = {{1, 2, 3, 4}, {4, 5, 6, 6}, {2, 5, 1, 2}, {7, 8, 9, 7}};
  auto [L, U, P] = LU_factorize(A);
  std::cout << " A\n" << A << std::endl;
  std::cout << " L\n" << L << std::endl;
  std::cout << " U\n" << U << std::endl;
  std::cout << " P\n" << P << std::endl;
  std::cout << " LU\n" << L * U << std::endl;
  std::cout << " PA\n" << P * A << std::endl;
}
