#ifndef PARALLEL_MATRIX_OPERATIONS_HPP
#define PARALLEL_MATRIX_OPERATIONS_HPP

#include <algorithm>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T, typename U, typename R = decltype(T() + U())>
std::vector<R> parallel_multiply(Matrix<T> const& m, std::vector<U> const& v)
{
  if (m[0].size() != v.size())
  {
    std::cerr
      << "ERROR: incorrectly sized matrix or vector in parallel mat * vec\n";
    exit(EXIT_FAILURE);
  }

  std::vector<R> result(m.size(), 0);

  auto i = 0u, j = 0u;
#pragma omp parallel
  {
#pragma omp for
    for (i = 0u; i < m.size(); ++i)
    {
      for (j = 0u; j < m[0].size(); ++j)
      {
        result[i] += m[i][j] * v[j];
      }
    }
  }

  return result;
}

template <typename T, typename U, typename R = decltype(T() + U())>
std::vector<R> parallel_multiply2(Matrix<T> const& m, std::vector<U> const& v)
{
  if (m[0].size() != v.size())
  {
    std::cerr
      << "ERROR: incorrectly sized matrix or vector in parallel mat * vec\n";
    exit(EXIT_FAILURE);
  }

  std::vector<R> result(m.size(), 0);

  auto i = 0u, j = 0u;
#pragma omp parallel
  {
#pragma omp for collapse(2)
    for (i = 0u; i < m.size(); ++i)
    {
      for (j = 0u; j < m[0].size(); ++j)
      {
        result[i] += m[i][j] * v[j];
      }
    }
  }

  return result;
}

template <typename T, typename U, typename R = decltype(T() + U())>
Matrix<R> parallel_multiply(Matrix<T> const& m1, Matrix<U> const& m2)
{
  if (m1[0].size() != m2.size())
  {
    std::cerr << "ERROR: incorrectly sized matrices in parallel mat * mat\n";
    exit(EXIT_FAILURE);
  }

  Matrix<R> result(m1.size());
  std::for_each(begin(result), end(result), [&m2](std::vector<R>& row) {
    row.resize(m2[0].size());
  });

#pragma omp parallel
  {
#pragma omp for
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
  }

  return result;
}

template <typename T, typename U, typename R = decltype(T() + U())>
Matrix<R> parallel_multiply2(Matrix<T> const& m1, Matrix<U> const& m2)
{
  if (m1[0].size() != m2.size())
  {
    std::cerr << "ERROR: incorrectly sized matrices in parallel mat * mat\n";
    exit(EXIT_FAILURE);
  }

  Matrix<R> result(m1.size());
  std::for_each(begin(result), end(result), [&m2](std::vector<R>& row) {
    row.resize(m2[0].size(), 0);
  });

#pragma omp parallel
  {
#pragma omp for collapse(3)
    for (auto i = 0u; i < result.size(); ++i)
    {
      for (auto j = 0u; j < result[0].size(); ++j)
      {
        for (auto k = 0u; k < m2.size(); ++k)
        {
          result[i][j] += m1[i][k] * m2[k][j];
        }
      }
    }
  }

  return result;
}
#endif
