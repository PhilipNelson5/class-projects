#ifndef EXPLICIT_EULER_HPP
#define EXPLICIT_EULER_HPP

#include "../machineEpsilon/maceps.hpp"
#include <iomanip>
#include <iostream>

template <typename T, typename F>
T predictor_corrector(T x0, T y0, T x, T dt, F f, unsigned int n, const unsigned int MAX_ITERATIONS = 1000)
{
  double A, B, ALPHA, H, t, W[MAX_ITERATIONS], K1, K2, K3, K4;

  A = 0.0;
  B = 2.0;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(10);

  H = (B - A) / n;
  t = A;
  W[0] = 0.5; // initial value

  for (auto i = 1u; i <= 3u; i++)
  {
    K1 = H * f(t, W[i - 1]);
    K2 = H * f(t + H / 2.0, W[i - 1] + K1 / 2.0);
    K3 = H * f(t + H / 2.0, W[i - 1] + K2 / 2.0);
    K4 = H * f(t + H, W[i - 1] + K3);

    W[i] = W[i - 1] + 1 / 6.0 * (K1 + 2.0 * K2 + 2.0 * K3 + K4);

    t = A + i * H;

    std::cout << "At time " << t << " the solution = " << W[i] << std::endl;
  }

  for (auto i = 4u; i <= n; i++)
  {
    K1 = 55.0 * f(t, W[i - 1]) - 59.0 * f(t - H, W[i - 2]) +
         37.0 * f(t - 2.0 * H, W[i - 3]) - 9.0 * f(t - 3.0 * H, W[i - 4]);
    W[i] = W[i - 1] + H / 24.0 * K1;

    t = A + i * H;

    std::cout << "At time " << t << " the solution = " << W[i] << std::endl;
  }

  return W[i];
}

#endif
