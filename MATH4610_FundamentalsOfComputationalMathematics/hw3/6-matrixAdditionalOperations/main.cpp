#include "../5-matrixOperations/matrixOperations.hpp"
#include "determinant.hpp"
#include "kroneckerProduct.hpp"
#include <iostream>
#include <vector>

int main()
{
  Matrix<double> m1 = {{1, 2}, {3, 4}, {1, 0}}, m2 = {{0, 5, 2}, {6, 7, 3}},
                 m3 = {{1, 3, 2}, {-3, -1, -3}, {2, 3, 1}};

  std::cout << m1 << '\n' << m2 << '\n' << m3 << '\n';

  std::cout << kronecker_product(m1, m2) << '\n';

  std::cout << determinant(m3) << '\n';
}
