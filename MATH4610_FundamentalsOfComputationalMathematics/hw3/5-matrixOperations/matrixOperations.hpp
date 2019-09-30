#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * Addition and Subtraction for std::vector<std::vector<T>>
 * used to represent a mathematics vector
 *
 * @tparam T Type of the elements in the first matrix
 * @tparam U Type of the elements in the second matrix
 * @tparam R Type of the elements in the result matrix
 * @param a  The first matrix
 * @param b  The second matrix
 * @return   The result of the addition or subtraction
 */
#define matrix_add_subtract(op)                                                \
  template <typename T, typename U, typename R = decltype(T() + U())>          \
  Matrix<R> operator op(Matrix<T> const& a, Matrix<U> const& b)                \
  {                                                                            \
    if (a.size() != b.size() || a[1].size() != b[0].size())                    \
    {                                                                          \
      std::cerr << "ERROR: bad size in matrix_add_subtract\n";                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
                                                                               \
    Matrix<R> result(a.size());                                                \
    for (auto i = 0u; i < a.size(); ++i)                                       \
    {                                                                          \
      result[i].reserve(a[i].size());                                          \
      for (auto j = 0u; j < a[i].size(); ++j)                                  \
      {                                                                        \
        result[i].push_back(a[i][j] op b[i][j]);                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    return result;                                                             \
  }

matrix_add_subtract(+) matrix_add_subtract(-)

  /**
   * Compute the transpose of a matrix
   *
   * @tparam T Type of the elements in the matrix
   * @param m  The matrix
   * @return   A new matrix which is the transpose of m
   */
  template <typename T>
  Matrix<T> transpose(Matrix<T> const& m)
{
  Matrix<T> tp(m[0].size());
  std::for_each(
    begin(tp), end(tp), [&](std::vector<T>& row) { row.resize(m.size()); });

  for (auto j = 0u; j < m[0].size(); ++j)
  {
    for (auto i = 0u; i < m.size(); ++i)
    {
      tp[j][i] = m[i][j];
    }
  }
  return tp;
}

/**
 * Compute the trace of a matrix
 *
 * @tparam T Type of the elements in the matrix
 * @param m  The matrix
 * @return   The trace of matrix m
 */
template <typename T>
T trace(Matrix<T> const& m)
{
  if (m.size() != m[0].size())
  {
    std::cerr << "ERROR: non square matrix in trace\n";
    exit(EXIT_FAILURE);
  }

  T t = 0;

  for (auto i = 0u; i < m.size(); ++i)
  {
    t += m[i][i];
  }
  return t;
}

/**
 * multiplication of a scalar and a matrix ( s * m )
 *
 * @tparam T Type of the elements in the matrix
 * @tparam S Type of the elements in the vector
 * @tparam R Type of the elements in the result vector
 * @param s  A scalar value
 * @param m  An mxn matrix
 * @return   A vector which holds the result of m * s
 */
template <typename T, typename S, typename R = decltype(S() * T())>
Matrix<R> operator*(S const s, Matrix<T> const& m)
{
  Matrix<R> result = m;

  std::for_each(std::begin(result), std::end(result), [&](auto& row) {
    std::for_each(std::begin(row), std::end(row), [&](auto& elem) {
      elem *= s;
    });
  });

  return result;
}

/**
 * multiplication of a matrix and a scalar ( m * s )
 *
 * @tparam T Type of the elements in the matrix
 * @tparam S Type of the elements in the vector
 * @tparam R Type of the elements in the result vector
 * @param m  An mxn matrix
 * @param s  A scalar value
 * @return   A vector which holds the result of m * s
 */
template <typename T, typename S, typename R = decltype(T() * S())>
inline Matrix<R> operator*(Matrix<T> const& m, S const s)
{
  return s * m;
}

/**
 * multiplication of a matrix and a vector ( m * v )
 *
 * @tparam T Type of the elements in the matrix
 * @tparam U Type of the elements in the vector
 * @tparam R Type of the elements in the result vector
 * @param m  An mxn matrix
 * @param v  A vector with n elements
 * @return   A vector which holds the result of m * v
 */

template <typename T, typename U, typename R = decltype(T() + U())>
std::vector<R> operator*(Matrix<T> const& m, std::vector<U> const& v)
{
  if (m[0].size() != v.size())
  {
    std::cerr << "ERROR: incorrectly sized matrix or vector in mat * vec\n";
    exit(EXIT_FAILURE);
  }
  std::vector<R> result(m.size());

  for (auto i = 0u; i < m.size(); ++i)
  {
    R sum = 0;
    for (auto j = 0u; j < v.size(); ++j)
    {
      sum += m[i][j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * multiplication of a matrix and a matrix ( m * v )
 *
 * @tparam T Type of the elements in the first matrix
 * @tparam U Type of the elements in the second matrix
 * @tparam R Type of the elements in the result matrix
 * @param m1 An nxm matrix
 * @param m2 An mxp matrix
 * @return   A matrix which holds the result of m1 * m2
 */
template <typename T, typename U, typename R = decltype(T() + U())>
Matrix<R> operator*(Matrix<T> const& m1, Matrix<U> const& m2)
{
  if (m1[0].size() != m2.size())
  {
    std::cerr << "ERROR: incorrectly sized matrices in mat * mat\n";
    exit(EXIT_FAILURE);
  }

  Matrix<R> result(m1.size());
  std::for_each(begin(result), end(result), [&m2](std::vector<R>& row) {
    row.resize(m2[0].size());
  });

  for (auto i = 0u; i < result.size(); ++i)
  {
    for (auto j = 0u; j < result[0].size(); ++j)
    {
      result[i][j] = 0;
      for (auto k = 0u; k < m2.size(); ++k)
      {
        result[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return result;
}

/**
 * A convenient way to print out the contents of a std::vector<std::vector<T>>
 *
 * @tparam T Type of the elements in the matrix
 * @param o  The ostream to put the matrix on
 * @param a  The matrix
 * @return   Return the stream so that the operator can be chained together
 */
template <typename T>
std::ostream& operator<<(std::ostream& o, Matrix<T> const& m)
{
  std::for_each(begin(m), end(m), [&o](std::vector<T> row) {
    o << "| ";
    std::for_each(begin(row), end(row), [&o](T e) {
      o << std::setw(10) << std::setprecision(3) << std::setfill(' ') << e;
    });
    o << " |\n";
  });

  return o;
}

#endif
