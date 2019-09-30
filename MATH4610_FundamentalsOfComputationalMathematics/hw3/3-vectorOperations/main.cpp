#include "vectorOperations.hpp"
#include <iostream>
#include <numeric>

int main()
{
  std::vector<double> a = {1.1, 2.3, 3.5};
  std::vector<double> b = {4.2, 5.4, 6.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';
  std::cout << "a + b\t" << a + b << '\n';
  std::cout << "a - b\t" << a - b << '\n';
  std::cout << "7.9 * a\t" << 7.9 * a << '\n';
  std::cout << "a · b\t" << inner_product(a, b) << "\n\n";
  std::cout << "a x b\t" << cross_product(a, b) << '\n';

  // std::array<unsigned int, 90000> c;
  // std::array<unsigned int, 90000> d;
  //
  // std::iota(begin(c), end(c), 0);
  // std::iota(begin(d), end(d), 1);
  //
  // auto res = c + d;
  // std::cout << std::accumulate(begin(res), end(res), 0) << std::endl;
}
