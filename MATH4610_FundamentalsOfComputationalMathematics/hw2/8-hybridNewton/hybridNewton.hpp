#ifndef HYBRID_NEWTON_HPP
#define HYBRID_NEWTON_HPP

#include "../6-newton/newton.hpp"
#include <tuple>
#include <utility>

template <typename T, typename F>
std::tuple<T, T> bisection_n(F f, T a, T b, T fa, const int n)
{
  T p, fp;
  for (auto i = 0; i < n; ++i)
  {
    p = (a + b) / 2;
    fp = f(p);

    if (fa * fp < 0)
    {
      b = p;
    }
    else
    {
      a = p;
      fa = fp;
    }
  }

  return {a, b};
}

/**
 * Hybrid method which takes advantage of the bisection method to reduce the
 * interval by ~one order of magnitude, then test newton's method for convergent
 * behavior. If Newton's method stays bounded, then the root is found with
 * newton's method. If Newton's method leaves the interval, then bisection is
 * used again to reduce the interval.
 *
 * @tparam T       The type of x0 and tolerance
 * @tparam F       A function of type T(T)
 * @tparam Fprime  A function of type T(T)
 * @param a        The lower bound of the interval
 * @param b        The upper bound of the interval
 * @param tol      The Tolerance
 * @param MAX_ITER The maximum iterations
 */
template <typename T, typename F, typename Fprime>
T root_finder_hybrid_newton(F f, Fprime fprime, T a, T b, T tol)
{
  if (a > b)
  {
    std::swap(a, b);
  }

  auto fa = f(a);
  auto fb = f(b);

  if (fa * fb > 0 || tol <= 0)
  {
    return {};
  }

  if (fa == 0) return a;
  if (fb == 0) return b;

  T x0;
  do
  {
    // four iterations of bisection to reduce interval
    // by ~one order of magnitude
    std::tie(a, b) = bisection_n(f, a, b, f(a), 4);

    // check newton's for convergent behavior
    x0 = root_finder_newton(f, fprime, (a + b) / 2, tol, 1);

  } while (a < x0 && x0 < b);

  return root_finder_newton(f, fprime, x0, tol);
}

#endif
