#include "../matrix/matrix.hpp"
#include "../matrix/matrix_util.hpp"
#include "../conjugateGradient/conjugateGradient.hpp"
#include <array>

int main()
{
  auto answer = solveFivePointStencil<double, 5>(0.0, 1.0, sin);
  auto finalMat = arrayToMat(answer);
  std::cout << "Answer in Matrix Form\n" << finalMat << std::endl;
}
