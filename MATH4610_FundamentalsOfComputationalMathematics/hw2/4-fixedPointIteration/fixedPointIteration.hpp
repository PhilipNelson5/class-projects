#ifndef FIXED_POINT_ITERATION_HPP
#define FIXED_POINT_ITERATION_HPP

#include <cmath>

/**
 * Implementation of functional iteration in order to find the root of
 * a function f(x) by iterating of a function g(x) such that g(x) = x
 *
 * @tparam T       The type of x0 and tol
 * @tparam G       A function of type T(T)
 * @param x0       The initial guess
 * @param tol      The tolerance
 * @param MAX_ITER The maximum number of iterations
 */
template <typename T, typename G>
T root_finder_fixed_point_iteration(G g, T x0, T tol, const int MAX_ITER = 100)
{
  T x1;
  for(auto i = 0.0; i < MAX_ITER; ++i)
  {
    x1 = g(x0);
    if(std::abs(x1-x0) < tol)
      return x1;
    x0 = x1;
  }
  return x0;
}

#endif
