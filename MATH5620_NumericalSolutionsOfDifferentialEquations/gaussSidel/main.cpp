#include <iostream>
#include "gaussSidel.hpp"

int main()
{
  Matrix<double, 2, 2> A ({
      {16, 3},
      {7, -11}
      });
  std::array<double, 2> b({{11, 13}});
  std::cout << "A\n" << A << "b\n" << b << '\n';
  std::cout << gauss_sidel(A, b, 100u) << "\n";
  std::cout << A.jacobiIteration(b, 100u) << "\n";

  Matrix<double, 4, 4> A1 ({
      {10, -1, 2, 0},
      {-1, 11, -1, 3},
      {2, -1, 10, -1},
      {0, 3, -1, 8}
      });
  std::array<double, 4> b1({{6, 25, -11, 15}});
  std::cout << "A\n" << A1 << "b\n" << b1 << '\n';
  std::cout << gauss_sidel(A1, b1, 100u) << "\n";
  std::cout << A1.jacobiIteration(b1, 100u) << "\n";
}
