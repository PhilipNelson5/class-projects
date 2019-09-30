#ifndef CHOLESKY_FACTORIZATION_HPP
#define CHOLESKY_FACTORIZATION_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
Matrix<T> cholesky_factorization(Matrix<T> const& A)
{
  if (A.size() != A[0].size())
  {
    std::cerr << "ERROR: non-square matrix in cholesky_factorization"
              << std::endl;
  }

  Matrix<T> L(A.size());
  std::for_each(
    std::begin(L), std::end(L), [&](auto& row) { row.resize(A.size()); });

  int n = A.size();
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < (i + 1); j++)
    {
      double s = 0;
      for (int k = 0; k < j; k++)
      {
        s += L[i][k] * L[j][k];
      }

      if (i == j)
      {
        L[i][j] = std::sqrt(A[i][i] - s);
      }
      else
      {
        L[i][j] = 1.0 / L[j][j] * (A[i][j] - s);
      }
    }
  }

  return L;
}

#endif
