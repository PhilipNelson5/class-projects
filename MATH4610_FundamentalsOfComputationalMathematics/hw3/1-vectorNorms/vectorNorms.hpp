#ifndef VECTOR_NORMS_HPP
#define VECTOR_NORMS_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

/* clang-format off */
/**
 * Determine the l_pNorm of a vector
 *
 * @tparam T The type of the elements in `a`
 * @tparam P The type of `p`
 * @param a  The vector
 * @param p  The `p` of the l_pNorm
 */
template <typename T, typename P>
inline T p_norm(std::vector<T> const& a, P const p)
{
  return std::pow(
      std::accumulate(
        begin(a), end(a), 0.0, [p](T acc, T const e) {
          return acc + std::pow(std::abs(e), p);
        }),
      1.0/p);
}

/**
 * Determine the l_pNorm of a vector
 *
 * @tparam T The type of the elements in `a`
 * @param a  The vector
 */
template <typename T>
inline T inf_norm(std::vector<T> const& a)
{
  return std::abs(
      *std::max_element(
        begin(a), end(a), [](T const a, T const b) {
          return std::abs(a) < std::abs(b);
        })
      );
}
/* clang-format on */

#endif
