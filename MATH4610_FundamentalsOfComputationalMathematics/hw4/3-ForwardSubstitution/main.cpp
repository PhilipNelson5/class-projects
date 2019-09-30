#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "forwardSubstitution.hpp"
#include <iostream>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

int main()
{
  Matrix<double> L({{1, 0, 0, 0}, {5, 1, 0, 0}, {4, -6, 1, 0}, {-4, 5, -9, 1}});

  std::vector<double> y{3, 5, -6, 8};

  auto b = L * y;

  std::cout << " L\n" << L << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual y\n" << y << std::endl;
  std::cout << " Calculated y\n";
  std::cout << forward_substitution(L, b) << std::endl;
}
