#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "random.hpp"
#include "matrix_util.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

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
  int findLargestInCol(unsigned int const& col, unsigned int const& row = 0)
  {
    T max = row;
    for (auto i = row + 1; i < M; ++i)
    {
      if (std::abs(m[i][col]) > std::abs(m[max][col])) max = i;
    }
    return max;
  }

  /* calculate the lower and upper triangles */
  std::tuple<Matrix<T, N, N>, Matrix<T, N, N>> luFactorize()
  {
    for (auto i = 0u; i < N; ++i)
    {
      auto pivot = findLargestInCol(i, i);
      if (pivot != i) swapRows(i, pivot);
      pivot = m[i][i];
      auto mod = identity<T, N>();
      for (auto j = i; j < N; ++j)
      {
        m[i][j] = -1 / m[i][j] * pivot;
      }
      m = mod * m;
    }
  }

private:
  std::array<std::array<T, N>, M> m;
};

#endif
