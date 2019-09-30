#ifndef OUTER_PRODUCT_HPP
#define OUTER_PRODUCT_HPP

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * Compute the Kronecker Product of two matrices
 *
 * @tparam T The type of the elements stored in m1 and m2
 * @param m1 The first matrix
 * @param m2 The second matrix
 * @return   the result of the Kronecker Product
 */
template <typename T>
Matrix<T> kronecker_product(Matrix<T> m1, Matrix<T> m2)
{
  // get the dimensionality of m1 and m2
  auto r1 = m1.size(), c1 = m1[0].size(), r2 = m2.size(), c2 = m2[0].size();

  // initialize the result matrix mr which is r1*r2 x c1*c2
  Matrix<T> mr(r1 * r2);
  std::for_each(std::begin(mr), std::end(mr), [&c1, c2](auto& row) {
    row.resize(c1 * c2);
  });

  // for each row in matrix 1
  for (auto i = 0u; i < r1; ++i)
  {
    // for each col in matrix 1
    for (auto j = 0u; j < c1; ++j)
    {
      // for each row in matrix 2
      for (auto k = 0u; k < r2; ++k)
      {
        // for each col in matrix 2
        for (auto l = 0u; l < c2; ++l)
        {
          // Each element of matrix m1 is
          // multiplied by the whole matrix m2
          // and stored in matrix mr
          mr[i * r2 + k][j * c2 + l] = m1[i][j] * m2[k][l];
        }
      }
    }
  }

  return mr;
}

#endif
