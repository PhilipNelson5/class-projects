#ifndef QUADRATIC_FORMULA_HPP
#define QUADRATIC_FORMULA_HPP

#include <array>
#include <cmath>
#include <optional>

template <typename T>
std::optional<std::array<T, 2>> quadratic_equation(T a, T b, T c)
{
  const auto descrim = (b * b) - (4.0 * a * c);
  if (descrim > 0)
  {
    const T r1 = (-b + sqrt(descrim)) / (2.0 * a);
    const T r2 = (-b - sqrt(descrim)) / (2.0 * a);

    return std::array {r1, r2};
  }
  return {};
}

#endif
