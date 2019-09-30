#ifndef GAUSSIAN_ELIMINATION_HPP
#define GAUSSIAN_ELIMINATION_HPP

#include "../../hw3/5-matrixOperations/matrixOperations.hpp"
#include <algorithm>
#include <cmath>

/**
 * @paramt T   Type of the elements in the matrix
 * @param mat  The matrix
 * @param ones Should the diagonal be ones
 */
template <typename T>
void gaussian_emlimination(Matrix<T>& mat, bool ones = false)
{
  int m = mat.size();    // row dimensionality
  int n = mat[0].size(); // col dimensionality
  int h = 0;             // row pivot
  int k = 0;             // col pivot

  while (h < m && k < n)
  {
    // find the next pivot
    auto piv = h;
    auto max = abs(mat[h][k]);
    for (auto i = h + 1; i < m; ++i)
    {
      if (max < abs(mat[i][k]))
      {
        max = abs(mat[i][k]);
        piv = i;
      }
    }

    if (mat[piv][k] == 0) // no pivot in the col
    {
      ++k; // go to next col
      continue;
    }

    std::swap(mat[h], mat[piv]); // swap pivot row with current row

    // for all rows below pivot
    for (auto i = h + 1; i < m; ++i)
    {
      auto f = mat[i][k] / mat[h][k];
      mat[i][k] = 0; // zero out the rest of the col

      // for all the rest of the elements in the row
      for (auto j = k + 1; j < n; ++j)
      {
        mat[i][j] = mat[i][j] - mat[h][j] * f;
      }
    }

    ++h, ++k; // increase current row and col
  }

  if (ones) // make ones down the diagonal
  {
    for (auto i = 0; i < m; ++i)
    {
      if (mat[i][i] == 0) break;
      for (auto j = i + 1; j < n; ++j)
      {
        mat[i][j] /= mat[i][i];
      }
      mat[i][i] = 1;
    }
  }
}

#endif
