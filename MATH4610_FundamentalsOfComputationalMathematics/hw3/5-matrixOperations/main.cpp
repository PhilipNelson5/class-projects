#include "../3-vectorOperations/vectorOperations.hpp"
#include "matrixOperations.hpp"
#include <iostream>
#include <vector>

int main()
{
  Matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<int> m2 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  std::vector<double> v1 = {1.1, 2.2, 3.3};
  std::vector<double> v2 = {1.1, 2.2, 3.3};
  std::cout << "m1\n";
  std::cout << m1 << std::endl;
  std::cout << "m2\n";
  std::cout << m2 << std::endl;
  std::cout << "v1\n";
  std::cout << v1 << std::endl;
  std::cout << "v2\n";
  std::cout << v2 << std::endl;

  std::cout << "m1+m2\n";
  std::cout << m1 + m1 << std::endl;
  std::cout << "m1-m2\n";
  std::cout << m1 - m1 << std::endl;

  std::cout << "transpose m1\n";
  std::cout << transpose(m1) << std::endl;

  std::cout << "transpose m2\n";
  std::cout << transpose(m2) << std::endl;

  std::cout << "trace m1\n";
  std::cout << trace(m1) << "\n\n";

  std::cout << "m1*3\n";
  std::cout << m1 * 3 << std::endl;
  std::cout << "3*m1\n";
  std::cout << 3 * m1 << std::endl;

  std::cout << "m1*v1\n";
  std::cout << m1 * v1 << "\n";

  std::cout << "m1*m2\n";
  std::cout << m1*m2 << std::endl;
}
