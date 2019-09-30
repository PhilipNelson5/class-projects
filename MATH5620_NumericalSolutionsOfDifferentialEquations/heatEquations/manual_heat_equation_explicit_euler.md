---
title: Heat Equations
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Heat Equation

**Routine Name:** heat_explicit_euler

**Author:** Philip Nelson

**Language:** C++

## Description

The heat equation is a linear partial differential equation

\\[\frac{\partial u}{\partial t}  = k \frac{\partial^2 u}{ \partial x^2 }\\]

that describes the distribution of heat (or variation in temperature) in a given region over time.

## Input
`heat_explicit_euler(T L, unsigned int Nx, T F, T a, T Tf)`

* `T L` - Length of the domain ([0,L])
* `unsigned int Nx` - The total number of mesh points
* `T F` - The dimensionless number \\(\alpha \cdot \frac{dt}{dx^2}\\), which implicitly specifies the time step
* `T a` - Variable coefficient (constant)
* `T Tf` - The stop time for the simulation

## Output

A vector containing the state of the system at \\(Tf\\).

## Code
{% highlight c++ %}
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
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  auto a = 0.7;
  auto F = 0.01;
  auto Nx = 10u;
  auto Tf = 1.0;
  auto L = 1.0;

  auto solution = heat_explicit_euler<double>(L, Nx, F, a, Tf);

  for(auto && x:solution)
    std::cout << x << '\n';

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
0
1.545
2.93876
4.04485
4.75501
4.99971
4.75501
4.04485
2.93876
1.545
0
```

**Last Modification date:** 5 May 2018
