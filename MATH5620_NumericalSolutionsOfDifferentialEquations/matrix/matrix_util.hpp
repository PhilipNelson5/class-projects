#ifndef MATRIX_UTIL_HPP
#define MATRIX_UTIL_HPP

#include "../machineEpsilon/maceps.hpp"
#include "random.hpp"
#include "vector_util.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <tuple>

struct Point
{
  Point(double x, double y) : x(x), y(y) {}
  double x;
  double y;
};

template <typename T, std::size_t M, std::size_t N>
class Matrix;

/* Dot Product row i of a and col j of b  */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          std::size_t O,
          typename R = decltype(T() * U())>
R dotProduct(Matrix<T, M, N> const& a,
             Matrix<U, N, O> const& b,
             unsigned int const& i = 0u,
             unsigned int const& j = 0u)
{
  R sum = 0;
  for (auto k = 0u; k < N; ++k)
  {
    sum += a.get(i, k) * b.get(k, j);
  }
  return sum;
}

/* Remove the nth Row  */
template <typename T, std::size_t M, std::size_t N>
Matrix<T, M - 1, N> removeRow(Matrix<T, M, N> const& a, unsigned int n)
{
  T matrix[M - 1][N];
  int loc = 0;
  for (auto i = 0u; i < M; ++i)
  {
    if (i == n) continue;
    for (auto j = 0u; j < N; ++j)
    {
      matrix[loc][j] = a.get(i, j);
    }
    ++loc;
  }
  return matrix;
}

/* Remove the nth Col  */
template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N - 1> removeCol(Matrix<T, M, N> const& a, unsigned int n)
{
  T matrix[M][N - 1];
  for (auto i = 0u; i < M; ++i)
  {
    int loc = 0;
    for (auto j = 0u; j < N; ++j)
    {
      if (j == n) continue;
      matrix[i][loc] = a.get(i, j);
      ++loc;
    }
  }
  return matrix;
}

/* Determinant */
template <typename T>
T determinant(Matrix<T, 2, 2> const& a)
{
  return a.get(0, 0) * a.get(1, 1) - a.get(0, 1) * a.get(1, 0);
}

/* Calculate Determinant  */
template <typename T, std::size_t N>
T determinant(Matrix<T, N, N> const& a)
{
  T det = 0;
  for (auto i = 0u; i < N; ++i)
  {
    auto val = a.get(0, i) * determinant(removeRow(removeCol(a, i), 0));
    if (i % 2)
      det -= val;
    else
      det += val;
  }
  return det;
}

/* Addition and Subtraction */
#define matrix_add_subtract(op)                                                \
  template <typename T,                                                        \
            typename U,                                                        \
            std::size_t M,                                                     \
            std::size_t N,                                                     \
            typename R = decltype(T() op U())>                                 \
  Matrix<R, M, N> operator op(                                                 \
    Matrix<T, M, N> const& a, Matrix<U, M, N> const& b)                        \
  {                                                                            \
    R matrix[M][N];                                                            \
    for (auto i = 0u; i < M; ++i)                                              \
    {                                                                          \
      for (auto j = 0u; j < N; ++j)                                            \
      {                                                                        \
        matrix[i][j] = a.get(i, j) op b.get(i, j);                             \
      }                                                                        \
    }                                                                          \
    return matrix;                                                             \
  }

matrix_add_subtract(+) matrix_add_subtract(-)

  /* Multiplication  Matrix * Matrix*/
  template <typename T,
            typename U,
            std::size_t M,
            std::size_t N,
            std::size_t O,
            typename R = decltype(T() * U())>
  Matrix<R, M, O> operator*(Matrix<T, M, N> const& a, Matrix<U, N, O> const& b)
{
  R matrix[M][O];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < O; ++j)
    {
      matrix[i][j] = dotProduct(a, b, i, j);
    }
  }
  return matrix;
}

/* Multiplication Matrix * Array */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
std::array<R, M> operator*(Matrix<T, M, N> const& a, std::array<U, N> const& b)
{
  std::array<R, M> matrix;
  for (auto i = 0u; i < M; ++i)
  {
    R sum = 0;
    for (auto j = 0u; j < N; ++j)
    {
      sum += a.get(i, j) * b[j];
    }
    matrix[i] = sum;
  }
  return matrix;
}

/* Multiplication Array * Matrix */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
std::array<R, M> operator*(std::array<U, N> const& b, Matrix<T, M, N> const& a)
{
  std::array<R, M> matrix;
  for (auto i = 0u; i < M; ++i)
  {
    R sum = 0;
    for (auto j = 0u; j < N; ++j)
    {
      sum += a.get(i, j) * b[j];
    }
    matrix[i] = sum;
  }
  return matrix;
}

/* Scalar Multiplication a * scalar */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
Matrix<R, M, N> operator*(Matrix<T, M, N> const& a, U const& scalar)
{
  R matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = (a.get(i, j) * scalar);
    }
  }
  return matrix;
}

/* Scalar Multiplication scalar*a */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
Matrix<R, M, N> operator*(U const& scalar, Matrix<T, M, N> const& a)
{
  R matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = (a.get(i, j) * scalar);
    }
  }
  return matrix;
}

/* Scalar Division a/scalar */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() / U())>
Matrix<R, M, N> operator/(Matrix<T, M, N> const& a, U const& scalar)
{
  R matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = (a.get(i, j) / scalar);
    }
  }
  return matrix;
}

/* Unary Minus */
template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator-(Matrix<T, M, N> const& a)
{
  T matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = -a.get(i, j);
    }
  }
  return matrix;
}

/* Equality Operator */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
bool operator==(Matrix<T, M, N> const& a, Matrix<U, M, N> const& b)
{
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      if (a.get(i, j) != b.get(i, j)) return false;
    }
  }
  return true;
}

/* Stream Insertion */
template <typename T, std::size_t M, std::size_t N>
std::ostream& operator<<(std::ostream& o, Matrix<T, M, N> const& m)
{
  for (auto i = 0u; i < M; ++i)
  {
    o << "| ";
    for (auto j = 0u; j < N; ++j)
    {
      o << std::setw(10) << std::setprecision(3) << std::setfill(' ')
        << m.get(i, j);
    }
    o << " |" << std::endl;
  }
  return o;
}

template <typename T, std::size_t M, std::size_t N>
T oneNorm(Matrix<T, M, N>& m)
{
  std::array<T, N> colSum;
  for (auto j = 0u; j < M; ++j)
  {
    colSum[j] = 0;
    for (auto i = 0u; i < N; ++i)
    {
      colSum[j] += std::abs(m[i][j]);
    }
  }

  return *std::max_element(colSum.begin(), colSum.end());
}

template <typename T, std::size_t M, std::size_t N>
T infNorm(Matrix<T, M, N>& m)
{
  std::array<T, N> rowSum;
  for (auto i = 0u; i < N; ++i)
    rowSum[i] = std::accumulate(m.begin(i), m.end(i), 0, [](T sum, T elem) {
      return sum + std::abs(elem);
    });

  return *std::max_element(rowSum.begin(), rowSum.end());
}

template <typename T, std::size_t N>
T powerIteration(Matrix<T, N, N> const& A, unsigned int const& MAX)
{
  std::array<T, N> b_k;

  for (auto&& e : b_k)
    e = randDouble(0.0, 10.0);

  for (auto i = 0u; i < MAX; ++i)
  {
    auto Ab_k = A * b_k;
    auto norm = pNorm(Ab_k, 2);
    b_k = Ab_k / norm;
  }
  return pNorm(A * b_k, 2);
}

template <typename T, std::size_t N>
T inversePowerIteration(Matrix<T, N, N>& A, unsigned int const& MAX)
{
  std::array<T, N> v;
  for (auto&& e : v)
    e = randDouble(0.0, 10.0);

  T lamda = 0;
  for (auto i = 0u; i < MAX; ++i)
  {
    auto w = A.solveLinearSystemLU(v);
    v = w / pNorm(w, 2);
    lamda = v * (A * v);
  }
  return lamda;
}

template <typename T, std::size_t N>
Matrix<T, N * N, N * N> fivePointStencil()
{
  return Matrix<T, N * N, N * N>([](int i, int j) {
    if (i == j) return -4;
    if (i + 1 == j) return i % 3 != 2 ? 1 : 0;
    if (i == j + 1) return i % 3 != 0 ? 1 : 0;
    if (i + 3 == j) return 1;
    if (i == j + 3) return 1;
    return 0;
  });
  // return {
  //   {-4, 1, 0, 1, 0, 0, 0, 0, 0},
  //   {1, -4, 1, 0, 1, 0, 0, 0, 0},
  //   {0, 1, -4, 0, 0, 1, 0, 0, 0},
  //   {1, 0, 0, -4, 1, 0, 1, 0, 0},
  //   {0, 1, 0, 1, -4, 1, 0, 1, 0},
  //   {0, 0, 1, 0, 1, -4, 0, 0, 1},
  //   {0, 0, 0, 1, 0, 0, -4, 1, 0},
  //   {0, 0, 0, 0, 1, 0, 1, -4, 1},
  //   {0, 0, 0, 0, 0, 1, 0, 1, -4}
  // };
}

template <typename T, std::size_t N>
Matrix<T, N * N, N * N> ninePointStencil()
{
  return Matrix<T, N * N, N * N>([](int i, int j) {
    if (i == j) return -20;
    if (i + 1 == j) return i % 3 != 2 ? 4 : 0;
    if (i == j + 1) return i % 3 != 0 ? 4 : 0;
    if (i + 2 == j) return i % 3 != 0 ? 1 : 0;
    if (i == j + 2) return i % 3 != 2 ? 1 : 0;
    ;
    if (i + 3 == j) return 4;
    if (i == j + 3) return 4;
    if (i + 4 == j) return i % 3 != 2 ? 1 : 0;
    if (i == j + 4) return i % 3 != 0 ? 1 : 0;
    ;
  });
  // return {
  //   {-4, 1, 0, 1, 0, 0, 0, 0, 0},
  //   {1, -4, 1, 0, 1, 0, 0, 0, 0},
  //   {0, 1, -4, 0, 0, 1, 0, 0, 0},
  //   {1, 0, 0, -4, 1, 0, 1, 0, 0},
  //   {0, 1, 0, 1, -4, 1, 0, 1, 0},
  //   {0, 0, 1, 0, 1, -4, 0, 0, 1},
  //   {0, 0, 0, 1, 0, 0, -4, 1, 0},
  //   {0, 0, 0, 0, 1, 0, 1, -4, 1},
  //   {0, 0, 0, 0, 0, 1, 0, 1, -4}
  // };
}

template <typename T, std::size_t N>
Matrix<T, N - 2, N - 2> generateMesh(T a, T b)
{
  auto h = (b - a) / (N - 1);

  return Matrix<T, N - 2, N - 2>(
    [&](int i, int j) { return (a + (i + 1) * h) * (a + (j + 1) * h); });
}

template <typename T, typename F, std::size_t N>
std::array<T, N * N> initMeshB(Matrix<T, N, N>& mesh, F f)
{
  std::array<T, N * N> b;
  for (auto i = 0u; i < N; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      b[i * N + j] = f(mesh[i][j]);
    }
  }
  return b;
}

template <typename T, std::size_t N, std::size_t M>
double conditionNumber(Matrix<T, N, M> m)
{
  auto max = powerIteration(m, 1000u);
  auto min = inversePowerIteration(m, 1000u);

  return max / min;
}

template <typename T, std::size_t N>
Matrix<T, int(std::sqrt(N)), int(std::sqrt(N))> arrayToMat(std::array<T, N> a)
{
  return Matrix<T, int(std::sqrt(N)), int(std::sqrt(N))>(
    [&](int i, int j) { return a[i * int(std::sqrt(N)) + j]; });
}

template <typename T, std::size_t N, typename F>
auto solveFivePointStencil(T a, T b, F f)
{
  auto mesh = generateMesh<double, N>(a, b);
  auto bv = initMeshB(mesh, f);
  auto stencil = fivePointStencil<double, N - 2>();
  auto res = stencil.solveLinearSystemLU(bv);
  return res;
}

template <typename T, std::size_t N, typename F>
auto solveNinePointStencil(T a, T b, F f)
{
  auto mesh = generateMesh<double, N>(a, b);
  auto bv = initMeshB(mesh, f);
  auto stencil = ninePointStencil<double, N - 2>();
  auto res = stencil.solveLinearSystemLU(bv);
  return res;
}

template <typename T, std::size_t M, size_t N>
Matrix<T, M, N> makeNDiag(std::vector<T> const& entries,
                          unsigned int const& offset)
{
  return Matrix<T, M, N>(
    [=](const unsigned int& row, const unsigned int& col) -> T {
      const int start = row + offset;
      const int end = start + entries.size() - 1;

      if ((int)col >= start && (int)col <= end)
      {
        return entries[(int)col - row - offset];
      }
      else
      {
        return 0;
      }
    });
}

#endif
