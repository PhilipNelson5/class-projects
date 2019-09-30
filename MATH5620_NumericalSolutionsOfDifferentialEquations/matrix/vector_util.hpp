#ifndef VECTOR_UTILL_HPP
#define VECTOR_UTILL_HPP

#include "random.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

template <typename T, std::size_t N>
std::array<T, N> initRandom(int start, int end)
{
  std::array<T, N> a;
  for (auto i = 0u; i < N; ++i)
    a[i] = randDouble(start, end);

  return a;
}

template <typename T, std::size_t M>
bool allclose(std::array<T, M> a, std::array<T, M> b, double tol)
{
  for (auto i = 0u; i < M; ++i)
    if (std::abs(a[i] - b[i]) > tol) return false;
  return true;
}

template <typename T, std::size_t N>
double pNorm(std::array<T, N> v, unsigned int const& p)
{
  double sum = 0.0;
  for (auto&& x : v)
    sum += std::pow(std::abs(x), p);
  return std::pow(sum, 1.0 / p);
}

template <typename T, std::size_t N>
double infNorm(std::array<T, N> v)
{
  T max = std::abs(v[0]);
  for (auto&& x : v)
    max = std::max(max, std::abs(x));
  return max;
}

template <typename T, typename U, std::size_t N>
bool operator==(std::array<T, N> a, std::array<U, N> b)
{
  for (auto i = 0u; i < N; ++i)
    if (a[i] != b[i]) return false;
  return true;
}

#define vector_add_subtract(op)                                                \
  template <typename T,                                                        \
            typename U,                                                        \
            typename R = decltype(T() op U()),                                 \
            std::size_t N>                                                     \
  std::array<R, N> operator op(                                                \
    std::array<T, N> const& a, std::array<U, N> const& b)                      \
  {                                                                            \
    std::array<R, N> result;                                                   \
    for (auto i = 0u; i < N; ++i)                                              \
      result[i] = a[i] op b[i];                                                \
    return result;                                                             \
  }

vector_add_subtract(+) vector_add_subtract(-)

#define vector_add_subtract_scalar(op)                                         \
  template <typename T,                                                        \
            typename U,                                                        \
            typename R = decltype(T() op U()),                                 \
            std::size_t N>                                                     \
  R operator op(std::array<T, N> const& a, U const& b)                         \
  {                                                                            \
    R result = 0;                                                              \
    for (auto i = 0u; i < N; ++i)                                              \
      result += a[i] op b;                                                     \
    return result;                                                             \
  }

  vector_add_subtract_scalar(+) vector_add_subtract_scalar(-)

#define vector_multiply_divide_scalar(op)                                      \
  template <typename T,                                                        \
            typename U,                                                        \
            typename R = decltype(T() op U()),                                 \
            std::size_t N>                                                     \
  std::array<R, N> operator op(std::array<T, N> const& a, U const& b)          \
  {                                                                            \
    std::array<R, N> result;                                                   \
    for (auto i = 0u; i < N; ++i)                                              \
      result[i] = a[i] op b;                                                   \
    return result;                                                             \
  }

    vector_multiply_divide_scalar(*) vector_multiply_divide_scalar(/)

#define scalar_multiply_divide_vector(op)                                      \
  template <typename T,                                                        \
            typename U,                                                        \
            typename R = decltype(T() op U()),                                 \
            std::size_t N>                                                     \
  std::array<R, N> operator op(U const& b, std::array<T, N> const& a)          \
  {                                                                            \
    std::array<R, N> result;                                                   \
    for (auto i = 0u; i < N; ++i)                                              \
      result[i] = b op a[i];                                                   \
    return result;                                                             \
  }

      scalar_multiply_divide_vector(*) scalar_multiply_divide_vector(/)

        template <typename T,
                  typename U,
                  typename R = decltype(T() * U()),
                  std::size_t N>
        R operator*(std::array<T, N> a, std::array<U, N> b)
{
  R result = 0;
  for (auto i = 0u; i < N; ++i)
    result += a[i] * b[i];
  return result;
}

// template <typename T, std::size_t M>
// std::ostream& operator<<(std::ostream& o, std::vector<T, M> const& a)
//{
// o << "[ ";
// for (auto i = 0u; i < M; ++i)
// std::for_each(begin(a), end(a), []() {
// o << std::setw(10) << std::setprecision(3) << std::setfill(' ') << a[i];
//});
// o << " ]" << std::endl;

// return o;
//}

#endif
