#include <cmath>

inline double logistic(double a, double b, double t, double p0)
{
  return a / (((a / p0) - b) * exp(-a * t) + b);
}
