#ifndef MATRIX_NORMS_HPP
#define MATRIX_NORMS_HPP

#include "../3-vectorOperations/vectorOperations.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * Determine the one norm for a matrix
 *
 * @tparam The type of the elements in the matrix
 * @param  The matrix to take the norm of
 */
template <typename T>
T one_norm(Matrix<T> m)
{
  std::vector<T> colSums;
  colSums.reserve(m[0].size());

  for (auto j = 0u; j < m[0].size(); ++j)
  {
    T sum = 0;
    for (auto i = 0u; i < m.size(); ++i)
    {
      sum += std::abs(m[i][j]);
    }
    colSums.push_back(sum);
  }

  return *std::max_element(begin(colSums), end(colSums));
}

/**
 * Determine the infinity norm for a matrix
 *
 * @tparam The type of the elements in the matrix
 * @param  The matrix to take the norm of
 */
template <typename T>
T inf_norm(Matrix<T> m)
{
  std::vector<T> rowSums;
  rowSums.reserve(m[0].size());

  std::for_each(begin(m), end(m), [&rowSums](auto const& row) {
    rowSums.push_back(std::accumulate(
      begin(row), end(row), 0.0, [](T acc, T e) { return acc + std::abs(e); }));
  });

  return *std::max_element(begin(rowSums), end(rowSums));
}

/**
 * Determine the infinity norm for a matrix
 *
 * @tparam The type of the elements in the matrix
 * @param  The matrix to take the norm of
 */
template <typename T>
double frobenius_norm(Matrix<T> m)
{
  double elems_squared = 0;

  std::for_each(begin(m), end(m), [&elems_squared](auto const& row) {
    std::for_each(begin(row), end(row), [&elems_squared](auto const& elem) {
      elems_squared += elem * elem;
    });
  });

  return std::sqrt(elems_squared);
}

#endif
