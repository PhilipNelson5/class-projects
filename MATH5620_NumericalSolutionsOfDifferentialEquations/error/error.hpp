#include <cmath>
#include <iostream>

double absoluteError(double approx, double value)
{
  return std::abs(value - approx);
}

double relativeError(double approx, double value)
{
  return std::abs(absoluteError(value, approx) / value);
}
