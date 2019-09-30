#ifndef NEWTON_HPP
#define NEWTON_HPP

#include <cmath>

/**
 * Implementation of Newton's method for root finding
 * on a function f with derivative fprime at an initial guess x0
 *
 * @tparam T       The type of x0 and tolerance
 * @tparam F       A function of type T(T)
 * @tparam Fprime  A function of type T(T)
 * @param x0       The starting point
 * @param tol      The Tolerance
 * @param MAX_ITER The maximum iterations
 */
template <typename T, typename F, typename Fprime>
T root_finder_newton(F f, Fprime fprime, T x0, T tol, const int MAX_ITER = 100)
{
  T x1;

  for (auto i = 0; i < MAX_ITER; ++i)
  {
    x1 = x0 - f(x0) / fprime(x0);
    if (std::abs(x1 - x0) < tol * std::abs(x1))
    {
      break;
    }
    x0 = x1;
  }

  return x1;
}

#endif
