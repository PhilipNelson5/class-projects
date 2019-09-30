#include "../3-vectorOperations/vectorOperations.hpp"
#include "../5-matrixOperations/matrixOperations.hpp"
#include "vectorOuterProduct.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
  std::vector<double> v1{1, 2, 3, 4};
  std::vector<double> v2{1, 2, 3, 4};
  std::cout << v1 << '\n' << v2 << '\n';
  std::cout << outer_product(v1, v2) << '\n';
}
