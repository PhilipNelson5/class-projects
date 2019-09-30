#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "backSubstitution.hpp"
#include <iostream>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  Matrix<double> U = {
    {3, 5, -6, 4}, {0, 4, -6, 9}, {0, 0, 3, 11}, {0, 0, 0, -9}};

  std::vector<double> x = {4, 6, -7, 9};
  auto b = U * x;

  std::cout << " U\n" << U << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual x\n" << x << std::endl;
  std::cout << " Calculated x\n";
  std::cout << back_substitution(U, b) << std::endl;
}
