#ifndef ERROR_HPP
#define ERROR_HPP

#include <cmath>

/**
 * calculates the absolute error
 * e_abs = | approx - value |
 *
 * @tparam T     The type of approx and value
 * @param approx The approximated value
 * @param value  The accurate value
 * @return       The absolute error
 */
template <typename T>
inline T absolute_error(const T approx, const T value)
{
  return std::abs(value - approx);
}

/**
 * calculates the relative error
 * e_rel = | (approx - value) / value |
 *
 * @tparam T     The type of approx and value
 * @param approx The approximated value
 * @param value  The accurate value
 * @param        The relative error
 */
template <typename T>
inline T relative_error(const T approx, const T value)
{
  return std::abs((value - approx) / value);
}

#endif
