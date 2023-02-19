#pragma once

#include <type_traits>
#include <random>

int random_range(const int low, const int high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(low, high);
  return dist(mt);
}

template <typename T>
T random_range(const T low, const T high)
{
  static_assert(std::is_floating_point<T>::value);
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<> dist(low, high);
  return dist(mt);
}

template <typename T>
void random_fill(T low, T high, std::vector<T> v)
{
  for (auto it = begin(v); it != end(v); ++it)
  {
    *it = random_range(low, high);
  }
}
