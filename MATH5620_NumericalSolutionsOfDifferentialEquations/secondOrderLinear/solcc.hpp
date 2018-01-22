#include <cmath>
#include <optional>
#include <complex>

/*
 * Second-Order Linear Constant Coefficients
 * ay'' + by' + cy = f(t)
 *
 */
template <typename N, typename A, typename B, typename C, typename T>
std::optional<std::complex<N>> solcc(N y0, N v0, A a, B b, C c, T t)
{
  // roots from the quadratic formula
  std::complex<N> const radical = (b * b) - (4.0 * a * c);
  auto const r1 = (-b + sqrt(radical)) / (2.0 * a);
  auto const r2 = (-b - sqrt(radical)) / (2.0 * a);

  // There is no solution if the roots are the same
  if (r1 == r2) return {};

  // calculate c1 and c2
  auto const c1 = (v0 - (r2 * y0)) / (r1 - r2);
  auto const c2 = ((r1 * y0) - v0) / (r1 - r2);

  // return the solution
  return c1 * exp(r1 * t) + c2 * exp(r2 * t);
}
