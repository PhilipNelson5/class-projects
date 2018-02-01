#ifndef MATRIX_UTIL_HPP
#define MATRIX_UTIL_HPP

#include <iomanip>
#include <iostream>
#include <tuple>

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

/* Addition */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() + U())>
Matrix<R, M, N> operator+(Matrix<T, M, N> const& a, Matrix<U, M, N> const& b)
{
  R matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = a.get(i, j) + b.get(i, j);
    }
  }
  return matrix;
}

/* Subtraction */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() - U())>
Matrix<R, M, N> operator-(Matrix<T, M, N> const& a, Matrix<U, M, N> const& b)
{
  R matrix[M][N];
  for (auto i = 0u; i < M; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      matrix[i][j] = a.get(i, j) - b.get(i, j);
    }
  }
  return matrix;
}

/* Multiplication */
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

/* Scalar Multiplication a*scalar */
template <typename T,
          typename U,
          std::size_t M,
          std::size_t N,
          typename R = decltype(T() * U())>
Matrix<R, M, N> operator*(Matrix<T, M, N> const& a, U scalar)
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
Matrix<R, M, N> operator*(U scalar, Matrix<T, M, N> const& a)
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
      o << std::setw(9) << std::setprecision(4) << std::setfill(' ') << m.get(i, j);
    }
    o << " |" << std::endl;
  }
  return o;
}

#endif
