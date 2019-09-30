#include "../1-vectorNorms/vectorNorms.hpp"
#include "../3-vectorOperations/vectorOperations.hpp"
#include "vectorError.hpp"
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

int main()
{
  std::vector<double> a = {1.1, 2.22, 3.5};
  std::vector<double> b = {1.2, 2.23, 3.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';

  std::cout << "absolute error of a and b\n"
            << absolute_error(
                 a, b, std::bind(p_norm<double, int>, std::placeholders::_1, 2))
            << std::endl;

  std::cout << "relative error of a and b\n"
            << relative_error(
                 a, b, std::bind(p_norm<double, int>, std::placeholders::_1, 2))
            << std::endl;
}
