#include "matrix.hpp"
#include <iostream>

int main()
{
  Matrix<int, 2, 3> a({{1, 2, 3}, {4, 5, 6}});
  Matrix<int, 2, 3> b({{1, 3, 5}, {5, 5, 6}});
  Matrix<double, 3, 2> c({{7, 8}, {9, 10}, {11, 12}});
  Matrix<double, 4, 4> d(0, 10);

  std::cout << a+b << std::endl;
  std::cout << a-b << std::endl;
  std::cout << a*c << std::endl;
  std::cout << determinant(d) << std::endl;
}
