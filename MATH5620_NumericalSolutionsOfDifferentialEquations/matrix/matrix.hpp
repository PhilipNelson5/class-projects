#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "matrix_util.hpp"
#include "random.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

template <typename T, std::size_t M, std::size_t N>
class Matrix;

/* returns an NxN identity matrix */
template <typename T, std::size_t N>
static Matrix<T, N, N> identity()
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
  using filler = std::function<T(unsigned int const&, unsigned int const&)>;
  /* Default Creation */
  Matrix() {}

  /* Filled Creation */
  Matrix(filler f)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = f(i, j);
  }

  /* Random Creation */
  Matrix(int start, int end)
  {
    for (auto i = 0u; i < M; ++i)
      for (auto j = 0u; j < N; ++j)
        m[i][j] = randInt(start, end);
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

  void set(unsigned int const& i, unsigned int const& j, T const& val)
  {
    m[i][j] = val;
  }

  std::array<T, N>& operator[](int x) { return m[x]; }

  auto begin(unsigned int n) { return m[n].begin(); }

  auto end(unsigned int n) { return m[n].end(); }

  /* Swap rows r1 and r2 */
  void swapRows(unsigned int const& r1, unsigned int const& r2)
  {
    std::swap(m[r1], m[r2]);
  }

  /* return the absolute largest element of a col starting at a given row */
  unsigned int findLargestInCol(unsigned int const& col,
                                unsigned int const& row = 0)
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

  /* calculate the lower and upper triangles with a permutation matrix*/
  std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>> luFactorize()
  {
    auto I = identity<T, N>();
    auto P = I;
    Matrix<T, N, N> L(0);
    Matrix<T, N, N> U(m);
    for (auto j = 0u; j < N; ++j) // columns
    {
      auto largest = U.findLargestInCol(j, j);
      if (largest != j)
      {
        L.swapRows(j, largest);
        U.swapRows(j, largest);
        P.swapRows(j, largest);
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

  std::array<T, M> backSub(std::array<T, M> b)
  {
    std::array<T, M> x;
    for (auto i = (int)M - 1; i >= 0; --i)
    {
      T sum = 0.0;
      for (auto j = (unsigned int)i + 1; j < M; ++j)
      {
        sum += m[i][j] * x[j];
      }
      x[i] = (b[i] - sum) / m[i][i];
    }
    return x;
  }

  std::array<T, M> forwardSub(std::array<T, M> b)
  {
    std::array<T, M> y;
    for (auto i = 0u; i < N; ++i)
    {
      T sum = 0.0;
      for (auto j = 0u; j < i; ++j)
      {
        sum += m[i][j] * y[j];
      }
      y[i] = b[i] - sum;
    }
    return y;
  }

  std::array<T, M> solveLinearSystemLU(std::array<T, M> b)
  {
    auto [L, U, P] = luFactorize();
    auto y = L.forwardSub(P * b);
    auto x = U.backSub(y);
    return x;
  }

  static std::array<T, M> triDiagThomas(std::array<T, M> const& a,
                                        std::array<T, M> const& b,
                                        std::array<T, M> const& c,
                                        std::array<T, M> const& d)
  {
    std::array<double, M> cp, dp, x;
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (auto i = 1u; i < N; ++i)
    {
      double bottom = (b[i] - (a[i] * cp[i - 1]));
      cp[i] = c[i] / bottom;
      dp[i] = (d[i] - (a[i] * dp[i - 1])) / bottom;
    }

    x[N - 1] = dp[N - 1];

    for (auto i = (int)N - 2; i >= 0; --i)
    {
      x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
  }

  std::array<T, M> triDiagThomas(std::array<T, M> const& d)
  {
    std::array<T, M> a, b, c;
    a[0] = 0;
    b[0] = m[0][0];
    c[0] = m[0][1];
    for (auto i = 1u; i < M - 1; ++i)
    {
      a[i] = m[i][i - 1];
      b[i] = m[i][i];
      c[i] = m[i][i + 1];
    }
    a[M - 1] = m[M - 1][M - 2];
    b[M - 1] = m[M - 1][M - 1];
    c[M - 1] = 0;

    return triDiagThomas(a, b, c, d);
  }

  Matrix<T, N * N, N * N> blockToMatrix(
    Matrix<Matrix<T, N, N>, N * N, N * N> block)
  {
    Matrix<T, N * N, N * N> res([&](int i, int j) {
      return block[i / N][j / N][i - (N * (i / N))][j - (N * (j / N))];
    });

    return res;
  }

private:
  std::array<std::array<T, N>, M> m;
};

#endif
