#include <cmath>
#include <iostream>

template <typename T>
inline T absoluteError(const T approx, const T value)
{
  return std::abs(value - approx);
}

template <typename T>
inline T relativeError(const T approx, const T value)
{
  return std::abs(absoluteError(value, approx) / value);
}
