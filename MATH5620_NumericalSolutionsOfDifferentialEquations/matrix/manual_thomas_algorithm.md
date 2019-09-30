---
title: Thomas Algorithm
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Thomas Algorithm

**Routine Name:** triDiagThomas

**Author:** Philip Nelson

**Language:** C++

## Description

`triDiagThomas` solves a tridiagonal linear system of equations using the Thomas Algorithm.

## Input

`triDiagThomas(std::array<T, M> const& d)` is called by a `Matrix<T, M, M>` of type `T` and size `M`x`M` and requires:


* `std::array<T, M> d` - an array of type `T` and size `M` (d vector)

## Output

`triDiagThomas` returns a `std::array<T, M>` with the solution vector `x`

## Code
{% highlight c++ %}
std::array<T, M> triDiagThomas(std::array<T, M> const& a,
                               std::array<T, M> const& b,
                               std::array<T, M> const& c,
                               std::array<T, M> const& d)
{
  std::array<double, M> cp, dp, x;
  cp[0] = c[0] / b[0];
  dp[0] = d[0] / b[0];
  for (auto i = 1u; i < N; ++i)
  {
    double bottom = (b[i] - (a[i] * cp[i - 1]));
    cp[i] = c[i] / bottom;
    dp[i] = (d[i] - (a[i] * dp[i - 1])) / bottom;
  }

  x[N - 1] = dp[N - 1];

  for (auto i = (int)N - 2; i >= 0; --i)
  {
    x[i] = dp[i] - cp[i] * x[i + 1];
  }
  return x;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 4, 4> A({
                         {-2,  1,  0,  0},
                         { 1, -2,  1,  0},
                         { 0,  1, -2,  1},
                         { 0,  0,  1, -2}
    });
  std::array<double, 4> x = {4, -3, 5, 1};

  auto d = A * x;
  auto testx = A.triDiagThomas(d);

  std::cout << " A\n" << A << std::endl;
  std::cout << " d\n" << d << std::endl;
  std::cout << " Real x\n" << x << std::endl;
  std::cout << " Calculated x\n" << testx << std::endl;
}
{% endhighlight %}

## Result
```
 A
|        -2        1        0        0 |
|         1       -2        1        0 |
|         0        1       -2        1 |
|         0        0        1       -2 |

 d
[       -11       15      -12        3 ]

 Real x
[         4       -3        5        1 ]

 Calculated x
[         4       -3        5        1 ]

```

**Last Modification date:** 07 February 2018
