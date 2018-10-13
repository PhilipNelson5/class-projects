#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <algorithm>
#include <functional>
#include <optional>
#include <random>

/**
 * Generate a random number from [low, high]
 *
 * @param low  The lower bound
 * @param high The upper bound
 * @return     A random number on the range [low, high]
 */
template <typename T>
int random_int(int low, int high, T seed)
{
  static std::mt19937 mt(seed);
  std::uniform_int_distribution<> dist(low, high);
  return dist(mt);
}

int random_int(int low, int high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(low, high);
  return dist(mt);
}

/**
 * Generate a random number from [low, high)
 *
 * @param low  The lower bound
 * @param high The upper bound
 * @return     A random number on the range [low, high)
 */
double random_double(double low, double high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<> dist(low, high);
  return dist(mt);
}

/**
 * Fill a container from [first, last) with random numbers from [low, high]
 *
 * @param first Iterator to beginning of range to fill
 * @param last  Iterator to end of range to fill
 * @param low   The lower bound
 * @param high  The upper bound
 */
template <typename it>
void random_int_fill(it first, it last, const int low, const int high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(low, high);
  std::generate(first, last, std::bind(dist, mt));
}

/**
 * Fill a container from [first, last) with random numbers from [low, high)
 *
 * @param first Iterator to beginning of range to fill
 * @param last  Iterator to end of range to fill
 * @param low   The lower bound
 * @param high  The upper bound
 */
template <typename it>
void random_double_fill(it first, it last, const double low, const double high)
{
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  std::generate(first, last, std::bind(dist, mt));
}

#endif
