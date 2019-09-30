#ifndef L_U_FACTORIZATION_HPP
#define L_U_FACTORIZATION_HPP

#include "../../utils/matrixUtils.hpp"
#include "../3-ForwardSubstitution/forwardSubstitution.hpp"
#include "../4-BackSubstitution/backSubstitution.hpp"
#include <iostream>
#include <tuple>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
unsigned int findLargestInCol(Matrix<T> const& m,
                              unsigned int const& row,
                              unsigned int const& col)
{
  T max = row;
  for (auto i = row + 1u; i < m.size(); ++i)
  {
    if (std::abs(m[i][col]) > std::abs(m[max][col]))
    {
      max = i;
    }
  }
  return max;
}

/**
 * @tparam T The type of the elements of A
 * @param m  The matrix to be decomposed
 * @return A tuple composed of the decomposed matrix m into 
 * L - Lower triangular
 * U - Upper triangular
 * P - Permutation matrix
 */
template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> LU_factorize(Matrix<T> m)
{
  if (m.size() != m[0].size())
  {
    std::cerr << "ERROR: Non square matrix in luFactorize\n";
    exit(EXIT_FAILURE);
  }

  auto N = m.size();

  auto I = identity<T>(N);
  auto P = I;

  Matrix<T> L = zeros<T>(N);
  Matrix<T> U = m;

  for (auto j = 0u; j < N; ++j) // columns
  {
    auto largest = findLargestInCol(U, j, j);

    if (largest != j)
    {
      std::swap(L[j], L[largest]);
      std::swap(U[j], U[largest]);
      std::swap(P[j], P[largest]);
    }

    auto pivot = U[j][j];
    auto mod = I;

    for (auto i = j + 1; i < N; ++i) // rows
    {
      mod[i][j] = -U[i][j] / pivot;
    }

    L = L + I - mod;
    U = mod * U;
  }

  L = I + L;

  return {L, U, P};
}

#endif
