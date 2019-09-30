---
title: Lax Wendroff
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Lax Wendroff

**Routine Name:** `lax_wendorf`

**Author:** Philip Nelson

**Language:** C++

## Description

`lax_wendorf` is a numerical method for the solution of hyperbolic partial differential equations, based on finite differences. It is second-order accurate in both space and time. This method is an example of explicit time integration where the function that defines governing equation is evaluated at the current time.[1](https://en.wikipedia.org/wiki/Lax–Wendroff_method)

## Input

{% highlight c++ %}
lax_wendorf(const T xDomain[],
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
std::vector<std::array<T, S - 2>> lax_wendorf(const T xDomain[],
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
  std::vector<double> coeffs = {a + b, 1 - 2 * b, -a + b};

  auto A = makeNDiag<double, S - 2, S - 2>(coeffs, -1u);

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

  auto solution = lax_wendorf<S, double>(xDomain, tDomain, dx, dt, eta, c);

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
         0          0   4.56e+03  -2.38e+04  -5.17e+05   4.78e+06    6.4e+07  -8.34e+08
         0       -675   3.87e+03   7.48e+04  -7.47e+05  -9.11e+06   1.28e+08   1.24e+09
       100       -624  -5.89e+03   9.07e+04   7.38e+05  -1.45e+07  -1.06e+08    2.4e+09
       100        100  -9.71e+03  -2.21e+04   1.41e+06   4.81e+06  -2.25e+08   -9.4e+08
       100        775  -3.77e+03  -1.13e+05   1.86e+05   1.81e+07   8.56e+06  -3.04e+09
         0        724   5.99e+03  -5.97e+04  -1.16e+06   5.26e+06   2.09e+08  -2.62e+08
         0          0   5.25e+03   4.61e+04  -6.66e+05  -1.11e+07   6.38e+07   2.11e+09
         0          0          0    3.8e+04   3.53e+05  -4.65e+06  -8.29e+07    4.2e+08
         0   4.56e+03  -2.38e+04  -5.17e+05   4.78e+06    6.4e+07  -8.34e+08  -8.83e+09
      -675   3.87e+03   7.48e+04  -7.47e+05  -9.11e+06   1.28e+08   1.24e+09  -2.16e+10
      -624  -5.89e+03   9.07e+04   7.38e+05  -1.45e+07  -1.06e+08    2.4e+09   1.66e+10
       100  -9.71e+03  -2.21e+04   1.41e+06   4.81e+06  -2.25e+08   -9.4e+08   3.74e+10
```

**Last Modification date:** 5 May 2018
