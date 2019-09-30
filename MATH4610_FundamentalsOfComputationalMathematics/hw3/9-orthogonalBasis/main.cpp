#include "../3-vectorOperations/vectorOperations.hpp"
#include "orthogonalBasis.hpp"
#include <iostream>
#include <tuple>
#include <vector>

int main()
{
  std::vector<double> v1 = {2, 5};
  std::vector<double> v2 = {6, 5};

  std::cout << "v1 " << v1 << '\n' << "v2 " << v2 << '\n';

  auto [u1, u2] = orthogonal_basis(v1, v2);
  std::cout << "u1 " << u1 << '\n' << "u2 " << u2 << std::endl;
  std::cout << "u1 · u2 " << inner_product(u1, u2) << std::endl;

  std::cout << "\n------------------------------\n" << std::endl;

  v1 = {1, 4};
  v2 = {-5, 2};

  std::cout << "v1 " << v1 << '\n' << "v2 " << v2 << '\n';

  std::tie(u1, u2) = orthogonal_basis(v1, v2);
  std::cout << "u1 " << u1 << '\n' << "u2 " << u2 << std::endl;
  std::cout << "u1 · u2 " << inner_product(u1, u2) << std::endl;
}
