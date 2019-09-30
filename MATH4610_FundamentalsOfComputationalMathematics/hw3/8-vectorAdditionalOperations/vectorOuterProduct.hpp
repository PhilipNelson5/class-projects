#ifndef VECTOR_OUTER_PRODUCT_HPP
#define VECTOR_OUTER_PRODUCT_HPP

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * The outer product of two vectors
 *
 * @tparam T The type of the elements in v1 and v2
 * @param v1 The first vector
 * @param v2 The second vector
 */
template <typename T>
Matrix<T> outer_product(std::vector<T> v1, std::vector<T> v2)
{
  // check that vectors are the same size
  if (v1.size() != v2.size())
  {
    std::cerr << "ERROR: bad size in Determinant\n";
    exit(EXIT_FAILURE);
  }

  // setup resultant matrix
  Matrix<T> m(v1.size());
  std::for_each(
    std::begin(m), std::end(m), [&](auto& row) { row.resize(v2.size()); });

  // m_{ij} = v1_i * v2_j
  for (auto i = 0u; i < v2.size(); ++i)
  {
    for (auto j = 0u; j < v1.size(); ++j)
    {
      m[i][j] = v1[i] * v2[j];
    }
  }

  return m;
}

#endif
