#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../utils/matrixUtils.hpp"
#include "../8-CholeskyFactorization/choleskyFactorization.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
  const auto n = 1000;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;

  auto start = std::chrono::high_resolution_clock::now();
  auto L = cholesky_factorization(A);
  auto end = std::chrono::high_resolution_clock::now();

  auto LLT = L * transpose(L);

  auto result = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Cholesky factorization completed in " << result << " ms"
            << std::endl;

  if (allclose(A, LLT, 1e-10))
    std::cout << "A == LLT" << std::endl;
  else
    std::cout << "A =/= LLT" << std::endl;
}
