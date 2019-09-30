#ifndef VECTOR_ERROR_HPP
#define VECTOR_ERROR_HPP

#include <cmath>
#include <vector>

/**
 * @tparam T     The type of the elements in approx and value
 * @tparam F     A function that takes a std::vector<T> and returns a T
 * @param approx The approximated vector
 * @param value  The accurate vector
 * @param norm   A function that takes a vector and returns a T
 * @return       The absolute error of the two vectors
 */
template <typename T, typename F>
inline T absolute_error(std::vector<T> const& approx,
                        std::vector<T> const& value,
                        F norm)
{
  return std::abs(norm(approx) - norm(value));
}

/**
 * @tparam T     The type of the elements in approx and value
 * @tparam F     A function that takes a std::vector<T> and returns a T
 * @param approx The approximated vector
 * @param value  The accurate vector
 * @param norm   A function that takes a vector and returns a T
 * @return       The relative error of the two vectors
 */
template <typename T, typename F>
inline T relative_error(std::vector<T> const& approx,
                        std::vector<T> const& value,
                        F norm)
{
  auto valNorm = norm(value);
  return std::abs((norm(approx) - valNorm) / valNorm);
}

#endif
