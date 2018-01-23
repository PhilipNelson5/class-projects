#include <cmath>

template <typename A, typename B, typename T, typename N>
inline N logistic(A a, B b, T t, N p0)
{
  return a / (((a-p0*b)/p0) * exp(-a * t) + b);
}
