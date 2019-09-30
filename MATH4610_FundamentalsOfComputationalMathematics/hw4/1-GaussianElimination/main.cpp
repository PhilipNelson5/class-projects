#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include "gaussianElimination.hpp"
#include <iostream>

int main()
{
   Matrix<double> m1 = {{1, -1, 2, -3}, {4, 4, -2, 1}, {-2, 2, -4, 6}};
  //Matrix<double> m1 = {
    //{1, -1, 2, -1}, {4, 4, -2, 1}, {-3, 5, -7, 12}, {-2, 2, -4, 4}};
  std::cout << "m1\n" << m1 << std::endl;
  gaussian_emlimination(m1);
  std::cout << m1 << std::endl;
}
