#ifndef DETERMINANT_HPP
#define DETERMINANT_HPP

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/* Remove the nth Row  */
template <typename T>
Matrix<T> removeRow(Matrix<T> const& a, unsigned int n)
{
  Matrix<T> m(a.size() - 1);
  std::for_each(
    std::begin(m), std::end(m), [&](auto& row) { row.resize(a[0].size()); });

  int row = 0;
  for (auto i = 0u; i < a.size(); ++i)
  {
    if (i == n) continue;
    m[row] = a[i];
    ++row;
  }
  return m;
}

/* Remove the nth Col  */
template <typename T>
Matrix<T> removeCol(Matrix<T> const& a, unsigned int n)
{
  Matrix<T> m(a.size());
  std::for_each(std::begin(m), std::end(m), [&](auto& row) {
    row.resize(a[0].size() - 1);
  });

  for (auto i = 0u; i < a.size(); ++i)
  {
    int col = 0;
    for (auto j = 0u; j < a[0].size(); ++j)
    {
      if (j == n) continue;
      m[i][col] = a[i][j];
      ++col;
    }
  }
  return m;
}

/**
 * Calculate the determinant of a matrix
 *
 * @tparam T The type of the elements in the matrix a
 * @param a  The matrix
 * @return   The determinant of a
 */
template <typename T>
T determinant(Matrix<T> const& a)
{
  // matrix must be square
  if (a.size() != a[0].size())
  {
    std::cerr << "ERROR: bad size in Determinant\n";
    exit(EXIT_FAILURE);
  }

  // base case, a 2x2 matrix
  if (a.size() == 2 && a[0].size() == 2)
  {
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
  }

  T det = 0;
  for (auto i = 0u; i < a.size(); ++i)
  {
    // find the determinant of the matrix removing row 0 and col i
    auto val = a[0][i] * determinant(removeRow(removeCol(a, i), 0));

    // subtract or add the value of
    // a[0][i] * the determinant of the sub-matrix
    if (i % 2)
      det -= val;
    else
      det += val;
  }

  return det;
}

#endif
