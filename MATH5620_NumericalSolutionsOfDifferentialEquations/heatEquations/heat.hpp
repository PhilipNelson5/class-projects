#ifndef HEAT_HPP_HPP
#define HEAT_HPP_HPP

#include <cmath>
#include <vector>

template <typename T>
std::vector<T> linspace(T start, T end, unsigned int n)
{
  std::vector<T> x;
  auto dx = (end - start) / n;
  for (auto i = 0u; i < n; ++i)
  {
    x.push_back(start + i * dx);
  }

  return x;
}

template <typename T>
auto heat_explicit_euler(T L, unsigned int Nx, T F, T a, T Tf)
{
  auto x = linspace<double>(0, L, Nx + 1);
  auto dx = x[1] - x[0];
  auto dt = F * dx * dx / a;
  auto Nt = std::round(Tf / dt);
  auto t = linspace<double>(0, Tf, Nt + 1);
  std::vector<T> u;
  std::vector<T> u_1;

  for (auto i = 0u; i <= Nx; ++i)
  {
    u.push_back(0);
    u_1.push_back(0);
  }

  u_1[Nx / 2] = 1000;

  for (auto n = 0u; n < Nt; ++n)
  {
    for (auto i = 1u; i < Nx; ++i)
    {
      u[i] = u_1[i] + F * (u_1[i - 1] - 2 * u_1[i] + u_1[i + 1]);
    }

    u[0] = 0;
    u[Nx] = 0;
    std::swap(u, u_1);
  }

  return u;
}

#endif
