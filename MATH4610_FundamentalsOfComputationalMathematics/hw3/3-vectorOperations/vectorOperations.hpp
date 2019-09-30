#ifndef VECTOR_OPERATIONS_HPP
#define VECTOR_OPERATIONS_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

/**
 * Addition and Subtraction for std::vector<T>
 * used to represent a mathematics vector
 *
 * @tparam T Type of the elements in the first vector
 * @tparam U Type of the elements in the second vector
 * @tparam R Type of the elements in the result vector
 * @param a  The first vector
 * @param b  The second vector
 * @return   The result of the addition or subtraction
 */
#define vector_add_subtract(op)                                                \
  template <typename T, typename U, typename R = decltype(T() op U())>         \
  std::vector<R> operator op(std::vector<T> const& a, std::vector<U> const& b) \
  {                                                                            \
    if (a.size() != b.size())                                                  \
    {                                                                          \
      std::cerr << "ERROR: bad size in vector addition\n";                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
                                                                               \
    std::vector<R> result(a.size());                                           \
    for (auto i = 0u; i < a.size(); ++i)                                       \
    {                                                                          \
      result[i] = a[i] op b[i];                                                \
    }                                                                          \
    return result;                                                             \
  }

vector_add_subtract(+) vector_add_subtract(-)

  /**
   * Multiplication of a scalar and a std::vector<T>
   * used to represent a mathematics vector
   *
   * @tparam T Type of the elements in the first vector
   * @tparam U Type of the scalar
   * @tparam R Type of the elements in the result vector
   * @param s  The scalar
   * @param a  The first vector
   * @return   The result of the scalar multiplication
   */
  template <typename T, typename U, typename R = decltype(T() * U())>
  std::vector<R> operator*(U const s, std::vector<T> const& a)
{
  std::vector<R> result(a.size());
  std::transform(
    std::begin(a), std::end(a), std::begin(result), [s](T e) { return e * s; });
  return result;
}

template <typename T, typename U, typename R = decltype(T() * U())>
std::vector<R> operator*(std::vector<T> const& v1, std::vector<U> const& v2)
{
  std::vector<R> result;
  result.reserve(v1.size());

  std::transform(std::begin(v1),
                 std::end(v1),
                 std::begin(v2),
                 std::back_inserter(result),
                 [](auto const& e1, auto const& e2) { return e1 * e2; });

  return result;
}

//template <typename T, typename U, typename R = decltype(T() / U())>
//std::vector<R> operator/(std::vector<T> const& a, std::vector<U> const b)
//{
  //std::vector<R> result(a.size());
  //for (auto i = 0u; i < a.size(); ++i)
    //result[i] = a[i] / b[i];
  //return result;
//}

template <typename T, typename U, typename R = decltype(T() / U())>
std::vector<R> operator/(std::vector<T> const& a, U const s)
{
  std::vector<R> result(a.size());
  std::transform(
    std::begin(a), std::end(a), std::begin(result), [s](T e) { return e / s; });
  return result;
}

/**
 * The inner product or dot product of two vectors
 *
 * @tparam T Type of the elements in the first vector
 * @tparam U Type of the elements in the second vector
 * @tparam R Type of the elements in the result vector
 * @param a  The first vector
 * @param b  The second vector
 * @return   The result of the inner product
 */
template <typename T, typename U, typename R = decltype(T() * U())>
R inner_product(std::vector<T> const& a, std::vector<U> const& b)
{
  if (a.size() != b.size())
  {
    std::cerr << "ERROR: bad size in vector inner product\n";
    exit(EXIT_FAILURE);
  }

  R product = 0.0;
  for (auto i = 0u; i < a.size(); ++i)
  {
    product += a[i] * b[i];
  }
  return product;
}

/**
 * The cross product of two vectors
 *
 * @tparam T Type of the elements in the first vector
 * @tparam U Type of the elements in the second vector
 * @tparam R Type of the elements in the result vector
 * @param a  The first vector
 * @param b  The second vector
 * @return   The result of the cross product
 */
template <typename T, typename U, typename R = decltype(T() * U())>
std::vector<R> cross_product(std::vector<T> const& a, std::vector<U> const& b)
{
  if (a.size() != 3 || b.size() != 3)
  {
    std::cerr << "ERROR: bad size in vector cross product\n";
    exit(EXIT_FAILURE);
  }

  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

/**
 * A convenient way to print out the contents of a std::vector<T>
 *
 * @tparam T Type of the elements in the vector
 * @param o  The ostream to put the vector on
 * @param a  The vector
 * @return   Return the stream so that the operator can be chained together
 */
template <typename T>
std::ostream& operator<<(std::ostream& o, std::vector<T> const& a)
{
  o << "[ ";
  std::for_each(begin(a), end(a), [&o](T e) {
    o << std::setw(10) << std::setprecision(3) << std::setfill(' ') << e;
  });
  o << " ]" << std::endl;

  return o;
}

#endif
