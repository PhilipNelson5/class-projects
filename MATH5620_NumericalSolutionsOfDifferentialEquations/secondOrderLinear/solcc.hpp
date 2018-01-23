#include <cmath>
#include <complex>
#include <optional>

/*
 * Second-Order Linear Constant Coefficients
 * ay'' + by' + cy = f(t)
 *
 */
template <typename T>
T solcc(T y0, T v0, T a, T b, T c, T t)
{
  // roots from the quadratic formula
  std::complex<T> const sqDiscrim = sqrt((b * b) - (4.0 * a * c));
  auto const r1 = (-b + sqDiscrim) / (2.0 * a);
  auto const r2 = (-b - sqDiscrim) / (2.0 * a);

  if (r1 == r2) // double roots
  {
    // calculate c1 and c2
    auto const c1 = y0;
    auto const c2 = v0 - r1 * y0;

    // return the solution
    return std::real(c1 * exp(r1 * t) + c2 * t * exp(r2 * t));
  }
  else // unique roots
  {
    // calculate c1 and c2
    auto const c1 = (v0 - (r2 * y0)) / (r1 - r2);
    auto const c2 = ((r1 * y0) - v0) / (r1 - r2);

    // return the solution
    return std::real(c1 * exp(r1 * t) + c2 * exp(r2 * t));
  }
}
