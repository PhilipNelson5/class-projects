#ifndef OUTER_PRODUCT_HPP
#define OUTER_PRODUCT_HPP

#include <algorithm>
#include <vector>

template <typename T>
void kronecker_product(std::vector<T> m1, std::vector<T> m2)
{
  auto r1 = m1.size(), c1 = m1[0].size(), r2 = m2.size(), c2 = m2[0].size();
  std::vector<T> mr(r1 * r2);
  std::for_each(
    std::begin(mr), std::end(mr), [](auto& row) { row.resize(c1 * c2); });

  for (int i = 0; i < r1; i++)
  {
    for (int k = 0; k < r2; k++)
    {
      for (int j = 0; j < c1; j++)
      {
        for (int l = 0; l < c2; l++)
        {
          // Each element of matrix m1 is
          // multiplied by whole Matrix m2
          // resp and stored as Matrix mr
          mr[i + l + 1][j + k + 1] = m1[i][j] * m2[k][l];
        }
      }
    }
  }
  return mr;
}

#endif
