#include <thrust/complex.h>

template <typename S, typename T, typename F, typename Fprime>
__device__
T root_finder_newton(F f, Fprime fprime, T x0, S tol, const int MAX_ITER = 100)
{
  T x1;

  for (auto i = 0; i < MAX_ITER; ++i)
  {
    x1 = x0 - f(x0) / fprime(x0);
    if (abs(x1 - x0) < tol * abs(x1))
    {
      break;
    }
    x0 = x1;
  }

  return x1;
}
