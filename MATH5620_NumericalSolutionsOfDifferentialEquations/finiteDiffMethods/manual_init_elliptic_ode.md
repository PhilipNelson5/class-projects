---
title: Init Elliptic ODE
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Initialize Elliptic ODE

**Routine Name:** initEllipticODE

**Author:** Philip Nelson

**Language:** C++

## Description

`initEllipticODE` sets up the vectors to solve the elliptic ODE

## Input

`initEllipticODE(F f, T a, T b, T ua, T ub)` requires:

* `F f` - the function \\(f\\)
* `T a` - the left boundary
* `T b` - the right boundary
* `T ua` - the value of \\(u(a)\\)
* `T ub` - the value of \\(u(a)\\)

## Output

`initEllipticODE` returns the `a b c` vectors for Thomas Algorithm, which are the diagonals, and the `f`, the function sampled along the mesh.

## Code
{% highlight c++ %}
template <std::size_t N, typename F, typename T>
std::tuple<std::array<T, N - 1>, std::array<T, N - 1>, std::array<T, N - 1>, std::array<T, N - 1>>
  initEllipticODE(F f, T a, T b, T ua, T ub)
{
  auto h = (b - a) / N;
  auto h2 = h * h;

  auto coeff = centralFinDiffCoeff<double, 2, 2>();

  std::array<T, N - 1> av, bv, cv, fv;

  av.fill(coeff[0]);
  bv.fill(coeff[1]);
  cv.fill(coeff[2]);

  for (auto i = 1u; i < N; ++i)
  {
    fv[i - 1] = h2 * f(a + i * h);
  }

  fv[0] -= ua;
  fv[fv.size() - 1] -= ub;

  return {av, bv, cv, fv};
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  constexpr int N = 10;
  auto a = 0.0, b = 1.0, ua = 2.5, ub = 5.0;
  auto h = (b - a) / N;

  auto[_a, _b, _c, _f] = initEllipticODE<N>([](double x) { return std::sin(M_PI * x); }, a, b, ua, ub);

  std::cout << "a\n" << _a << std::endl;
  std::cout << "b\n" << _b << std::endl;
  std::cout << "c\n" << _c << std::endl;
  std::cout << "f\n" << _f << std::endl;
}
{% endhighlight %}

## Result
```
a
[          1         1         1         1         1         1         1         1         1 ]

b
[         -2        -2        -2        -2        -2        -2        -2        -2        -2 ]

c
[          1         1         1         1         1         1         1         1         1 ]

f
[       -2.5   0.00588   0.00809   0.00951      0.01   0.00951   0.00809   0.00588        -5 ]

```

**Last Modification date:** 08 February 2018
