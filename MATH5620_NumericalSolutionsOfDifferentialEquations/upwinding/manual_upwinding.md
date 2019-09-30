---
title: Upwinding
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Upwinding

**Routine Name:** `upwinding`

**Author:** Philip Nelson

**Language:** C++

## Description

`upwinding` is in a class of numerical discretization methods for solving hyperbolic partial differential equations. Upwind schemes use an adaptive or solution-sensitive finite difference stencil to numerically simulate the direction of propagation of information in a flow field. The upwind schemes attempt to discretize hyperbolic partial differential equations by using differencing biased in the direction determined by the sign of the characteristic speeds. Historically, the origin of upwind methods can be traced back to the work of Courant, Isaacson, and Rees who proposed the CIR method.[1](https://en.wikipedia.org/wiki/Upwind_scheme)

## Input

{% highlight c++ %}
upwinding(const T xDomain[],
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
std::vector<std::array<T, S - 2>> upwinding(const T xDomain[],
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

  std::vector<double> coeffs = {c * dx / dt, 1 - c * dx / dt};

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
  constexpr auto dt = .1; // for view able output
  constexpr auto dx = .1; // for view able output
  constexpr std::size_t S = ((xDomain[1] - xDomain[0]) / dx);
  auto c = 0.7;
  auto eta = [](const double& x) -> double {
    return (x >= 0.3 && x <= 0.6) ? 100 : 0;
  };

  auto solution = upwinding<S, double>(xDomain, tDomain, dx, dt, eta, c);

  for (auto i = 0u; i < solution.size(); ++i)
  {
    for (auto j = 0u; j < solution[i].size(); ++j)
    {
      std::cout << std::setprecision(3) << std::setw(7) << solution[j][i]
    }
    std::cout << '\n';
  }

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
      0       0       0       0       0       0       0       0
      0       0       0       0       0       0       0       0
    100      30       9     2.7    0.81   0.243  0.0729  0.0219
    100     100      51    21.6    8.37    3.08    1.09   0.379
    100     100     100    65.7    34.8    16.3    7.05    2.88
      0      70      91    97.3    75.2    46.9    25.5    12.6
      0       0      49    78.4    91.6    80.1    56.9    34.9
      0       0       0    34.3    65.2    83.7    81.2    64.2
      0       0       0       0       0       0       0       0
      0       0       0       0       0       0       0       0
     30       9     2.7    0.81   0.243  0.0729  0.0219 0.00656
    100      51    21.6    8.37    3.08    1.09   0.379   0.129

```

**Last Modification date:** 5 May 2018
