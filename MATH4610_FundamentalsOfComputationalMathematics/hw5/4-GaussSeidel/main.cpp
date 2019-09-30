#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../2-MatrixGenerator/matrixGenerator.hpp"
#include "gaussSeidel.hpp"
#include <iostream>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(4u);
  auto b = generate_right_side(A);
  auto x = gauss_seidel(A, b);
  auto Ax = A * x;

  std::cout << " A\n" << A << std::endl;
  std::cout << " x\n" << x << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " A * x\n" << Ax << std::endl;
}
