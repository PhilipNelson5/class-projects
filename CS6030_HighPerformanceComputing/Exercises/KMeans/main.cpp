#include <iostream>
#include "KMeans.hpp"
#include "../../lib/type.hpp"
#include "../../lib/print.hpp"

int main()
{
  std::vector<int> v;
  v.resize(1);
  random_fill(0, 10, begin(v), end(v));
  std::cout << v << std::endl;
}