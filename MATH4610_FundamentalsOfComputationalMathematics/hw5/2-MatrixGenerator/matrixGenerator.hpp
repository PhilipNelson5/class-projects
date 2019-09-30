#ifndef MATRIX_GENERATOR_HPP
#define MATRIX_GENERATOR_HPP

#include "random.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

Matrix<double> generate_square_symmetric_diagonally_dominant_matrix(
  unsigned int const n)
{
  // initialize the matrix
  Matrix<double> m(n);
  std::for_each(std::begin(m), std::end(m), [&](auto& row) { row.resize(n); });

  // initialize the upper triangle
  for (auto i = 0u; i < n; ++i)
  {
    for (auto j = i; j < n; ++j)
    {
      m[i][j] = random_double(-1e1, 1e1);
    }
  }

  // copy to the lower triangle
  for (auto i = 0u; i < n; ++i)
  {
    for (auto j = 0u; j < i; ++j)
    {
      m[i][j] = m[j][i];
    }
  }

  // enforce diagonal dominance
  for (auto i = 0u; i < n; ++i)
  {
    auto amax = *std::max_element(
      std::begin(m[i]), std::end(m[i]), [](auto const& e1, auto const& e2) {
        return std::abs(e1) < std::abs(e2);
      });

    if (amax != m[i][i])
    {
      m[i][i] = (amax > 0 ? amax + random_double(0, 1e1)
                          : amax - random_double(0, 1e1));
    }
  }

  return m;
}

template <typename T>
inline std::vector<T> generate_right_side(Matrix<T> m)
{
  return m * std::vector<T>(m.size(), 1);
}

#endif
