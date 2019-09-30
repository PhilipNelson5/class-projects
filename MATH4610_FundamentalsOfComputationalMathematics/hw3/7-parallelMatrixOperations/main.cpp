#include "../3-vectorOperations/vectorOperations.hpp"
#include "../5-matrixOperations/matrixOperations.hpp"
#include "parallelMatrixOperations.hpp"
#include "random.hpp"
#include <chrono>
#include <iostream>
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename F>
auto timef(F f)
{
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(end - start).count();
}

void testMVM()
{
  auto trials = 25u;
  auto M = 10u, N = 10u;
  for (; N < 10000u; M += 100, N += 100)
  {
    double dur1 = 0, dur2 = 0, dur3 = 0;
    for (auto i = 0u; i < trials; ++i)
    {
      Matrix<int> m(M);
      std::for_each(std::begin(m), std::end(m), [&](auto& row) {
        row.reserve(M);
        for (auto i = 0u; i < N; ++i)
        {
          row.push_back(random_double(-1000000, 1000000));
        }
      });

      std::vector<double> v;
      for (auto i = 0u; i < N; ++i)
      {
        v.push_back(random_double(-1000000, 1000000));
      }

      dur1 += timef([&]() { m* v; });
      dur2 += timef([&]() { parallel_multiply(m, v); });
      dur3 += timef([&]() { parallel_multiply2(m, v); });
    }
    std::cout << M * N << " " << dur1 / trials << " " << dur2 / trials << " "
              << dur3 / trials << std::endl;
  }
}

void testMMM()
{
  auto trials = 25u;
  auto M = 10u, N = 10u;
  for (; N < 10000u; M += 25, N += 25)
  {
    Matrix<int> m(M);
    std::for_each(std::begin(m), std::end(m), [&](auto& row) {
      row.reserve(M);
      for (auto i = 0u; i < N; ++i)
      {
        row.push_back(random_double(-1000000, 1000000));
      }
    });

    double dur1 = 0, dur2 = 0, dur3 = 0;
    for (auto i = 0u; i < trials; ++i)
    {
      dur1 += timef([&]() { m* m; });
      dur2 += timef([&]() { parallel_multiply(m, m); });
      dur3 += timef([&]() { parallel_multiply2(m, m); });
    }
    std::cout << M * N << " " << dur1 / trials << " " << dur2 / trials << " "
              << dur3 / trials << std::endl;
  }
}

int main()
{
  testMVM();
}
