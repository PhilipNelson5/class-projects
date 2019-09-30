#include "vectorNorms.hpp"
#include "../3-vectorOperations/vectorOperations.hpp"
#include <iostream>
#include <vector>

int main()
{
  std::vector<double> v{3, 4, 1};
  std::cout << "v : " << v << '\n';
  std::cout << "l_1 norm :   " <<  l_pNorm(v, 1.0) << '\n';
  std::cout << "l_2 norm :   " << l_pNorm(v, 2.0) << '\n';
  std::cout << "l_inf norm : " <<  l_inf(v) << '\n';
}
