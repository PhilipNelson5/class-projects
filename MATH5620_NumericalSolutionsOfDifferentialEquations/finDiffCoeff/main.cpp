#include "finDiffCoeff.hpp"
#include <iostream>
#include "../matrix/matrix.hpp"
#include "../matrix/matrix_util.hpp"
#include "../matrix/vector_util.hpp"

int main()
{
  auto coeffs = centralFinDiffCoeff<double, 2, 4>();

  std::cout << "coefficients of a second order derivative with 4th accuracy\n";
  std::cout << coeffs << std::endl;

  return EXIT_SUCCESS;
}
