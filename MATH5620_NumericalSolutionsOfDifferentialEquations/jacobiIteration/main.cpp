#include "../matrix/matrix.hpp"
#include "jacobiIteration.hpp"
#include <array>

int main()
{
  Matrix<double, 4, 4> A({
      {-2,  1,  0,  0},
      { 1, -2,  1,  0},
      { 0,  1, -2,  1},
      { 0,  0,  1, -2}
      });
  std::array<double, 4> x = {
      {4, 7, 2, 5}
      };
  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;
  std::cout << jacobiIteration(A, b, 1000u) << std::endl;
}
