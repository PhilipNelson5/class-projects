#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "matrix_util.hpp"
#include "random.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

template <typename T, std::size_t M, std::size_t N>
class Matrix;

// template <typename T>
// using filler = std::function<T(unsigned int const&, unsigned int const&)>;

/* returns an NxN identity matrix */
template <typename T, std::size_t N>
Matrix<T, N, N> identity()
{
  Matrix<T, N, N> matrix(0);
  for (auto i = 0u; i < N; ++i)
  {
    matrix.set(i, i, 1);
  }
  return matrix;
}

template <typename T, std::size_t M, std::size_t N>
class Matrix
{
public:
  /* Default Creation */
  Matrix() {}

  /* Random Creation */
  Matrix(int start, int end)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = rand(start, end);
  }

  /* Fill With n */
  Matrix(int n)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = n;
  }

  /* Construct From Vector */
  Matrix(std::vector<std::vector<T>> v)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = v[i][j];
  }

  /* Construct From std::array */
  Matrix(std::array<std::array<T, N>, M> a)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = a[i][j];
  }

  /* Construct From Array */
  Matrix(T t[M][N])
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = t[i][j];
  }

  /* Copy Constructor */
  Matrix(Matrix const& old)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = old.m[i][j];
  }

  T get(unsigned int const& i, unsigned int const& j) const { return m[i][j]; }

  void set(unsigned int const& i, unsigned int const& j, T const& val) { m[i][j] = val; }

  std::array<T, N>& operator[](int x) { return m[x]; }

  /* Swap rows r1 and r2 */
  void swapRows(unsigned int const& r1, unsigned int const& r2)
  {
    for (auto i = 0u; i < N; ++i)
    {
      std::swap(m[r1][i], m[r2][i]);
    }
    // return this;
  }

  /* return the absolute largest element of a col starting at a given row */
  unsigned int findLargestInCol(unsigned int const& col, unsigned int const& row = 0)
  {
    T max = row;
    for (auto i = row + 1; i < M; ++i)
    {
      if (std::abs(m[i][col]) > std::abs(m[max][col])) max = i;
    }
    return max;
  }

  void transpose()
  {
    for (auto i = 0u; i < M; i++)
      for (auto j = 0u; j < N; j++)
        std::swap(m[j][i], m[i][j]);
  }

  /* calculate the lower and upper triangles */
  std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>> luFactorize()
  {
    auto I = identity<T, N>();
    auto P = identity<T, N>();
    P.transpose();
    Matrix<T, N, N> L(0);
    Matrix<T, N, N> U(m);
    std::vector<std::vector<unsigned int>> swaps;
    for (auto j = 0u; j < N; ++j) // columns
    {
      auto largest = U.findLargestInCol(j, j);
      if (largest != j)
      {
        L.swapRows(j, largest);
        U.swapRows(j, largest);
        P.swapRows(j, largest);
        swaps.push_back({j, largest});
      }
      auto pivot = U[j][j];
      auto mod = identity<T, N>();
      for (auto i = j + 1; i < N; ++i) // rows
      {
        mod[i][j] = -U[i][j] / pivot;
      }
      L = -(mod - I) + L;
      U = mod * U;
    }
    L = I + L;
    return {L, U, P};
  }

private:
  std::array<std::array<T, N>, M> m;
};

#endif
