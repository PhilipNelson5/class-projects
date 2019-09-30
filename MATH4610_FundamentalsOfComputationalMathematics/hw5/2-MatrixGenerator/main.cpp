#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "../../hw3/3-vectorOperations/vectorOperations.hpp"
#include "matrixGenerator.hpp"
#include <iostream>

int main()
{
  auto m = generate_square_symmetric_diagonally_dominant_matrix(5);
  auto b = generate_right_side(m);
  std::cout << " M\n" << m << std::endl;
  std::cout << " b\n" << b << std::endl;
}
