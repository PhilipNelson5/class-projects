---
title: Warming and Beam
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Warming and Beam

**Routine Name:** `warming_and_beam`

**Author:** Philip Nelson

**Language:** C++

## Description

`warming_and_beam` introduced in 1978 by Richard M. Beam and R. F. Warming, is a second order accurate implicit scheme, mainly used for solving non-linear hyperbolic equation. **It is not used much nowadays**.[1](https://en.wikipedia.org/wiki/Lax–Wendroff_method)

## Input

{% highlight c++ %}
warming_and_beam(const T xDomain[],
                 const T tDomain[],
                 const T dx,
                 const T dt,
                 F eta,
                 const T c
                )
{% endhighlight %}

* `T xDomain[]` - two member array with the spacial bounds
* `T tDomain[]` - two member array with the temporal bounds
* `T dx` - the spacial step
* `T dt` - the temporal step
* `F eta` - the function eta
* `T c` - the constant

## Output

A `std::vector<std::array<T, S - 2>>` with the solution over time.

## Code
{% highlight c++ %}
template <std::size_t S, typename T, typename F>
std::vector<std::array<T, S - 2>> warming_and_beam(const T xDomain[],
                                            const T tDomain[],
                                            const T dx,
                                            const T dt,
                                            F eta,
                                            const T c)
{
  std::array<T, S - 2> U;
  for (auto i = 0u; i < S - 2; ++i)
  {
    auto A = xDomain[0];
    auto x = A + (i + 1) * dx;
    U[i] = eta(x);
  }

  auto a = c * dx / dt / dt;
  auto b = c * c * dx * dx / dt / dt / 2;
  std::vector<double> coeffs = {b - a, 4 * a - 2 * b, 1 - 2 * a + b};

  auto A = makeNDiag<double, S - 2, S - 2>(coeffs, -2u);

  std::vector<std::array<T, S - 2>> solution = {U};

  for (auto t = tDomain[0]; t < tDomain[1]; t += dt)
  {
    U = A * U;
    solution.push_back(U);
  }

  return solution;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  constexpr double xDomain[] = {0.0, 1.0};
  constexpr double tDomain[] = {0.0, 1.0};
  // constexpr auto dt = 1.0e-3;
  // constexpr auto dx = 1.0e-3;
  constexpr auto dt = .1;
  constexpr auto dx = .1;
  constexpr std::size_t S = ((xDomain[1] - xDomain[0]) / dx);
  auto c = 0.7;
  auto eta = [](const double& x) -> double {
    return (x >= 0.3 && x <= 0.6) ? 100 : 0;
  };

  auto solution = warming_and_beam<S, double>(xDomain, tDomain, dx, dt, eta, c);

  for (auto i = 0u; i < solution.size(); ++i)
  {
    for (auto j = 0u; j < solution[i].size(); ++j)
    {
      std::cout << std::setprecision(3) << std::setw(10) << solution[j][i]
                << " ";
    }
    std::cout << '\n';
  }

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
         0          0          0          0          0          0          0          0
         0          0          0          0          0          0          0          0
       100  -1.28e+03   1.63e+04  -2.08e+05   2.65e+06  -3.38e+07   4.31e+08  -5.49e+09
       100   1.48e+03  -5.39e+04   1.14e+06  -2.02e+07    3.3e+08  -5.14e+09   7.74e+10
       100        800    3.9e+04  -2.09e+06   5.93e+07  -1.33e+09   2.63e+10  -4.79e+11
         0   2.08e+03  -1.44e+04   1.62e+06  -8.59e+07   2.86e+09  -7.53e+10   1.72e+12
         0       -675   6.03e+04  -1.43e+06    7.7e+07  -3.74e+09   1.35e+11  -3.98e+12
         0          0  -3.26e+04   2.17e+06   -7.8e+07   3.69e+09  -1.69e+11    6.4e+12
         0          0          0          0          0          0          0          0
         0          0          0          0          0          0          0          0
 -1.28e+03   1.63e+04  -2.08e+05   2.65e+06  -3.38e+07   4.31e+08  -5.49e+09   7.01e+10
  1.48e+03  -5.39e+04   1.14e+06  -2.02e+07    3.3e+08  -5.14e+09   7.74e+10  -1.14e+12
```

**Last Modification date:** 5 May 2018
