---
title: Solve Elliptic ODE K
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Solve Elliptic ODE with K

**Routine Name:** solveEllipticODEwithK

**Author:** Philip Nelson

**Language:** C++

## Description

`solveEllipticODEwithK` uses the `initEllipticODE` function and solves the elliptic ODE

## Input

`solveEllipticODEwithK(F f, std::array<T, N> k, T a, T b, T ua, T ub)` requires:

* `F f` - the function \\(f\\)
* `std::array<T, N> k` - vector for function \\(k\\)
* `T a` - the left boundary
* `T b` - the right boundary
* `T ua` - the value of \\(u(a)\\)
* `T ub` - the value of \\(u(a)\\)

## Output

`solveEllipticODEwithK` returns an array of approximations of the elliptic ODE at discrete points from \\(a\\) to \\(b\\)

## Code
{% highlight c++ %}
template <std::size_t N, typename F, typename T>
std::array<T, N - 1> solveEllipticODEwithK(F f, T a, T b, T ua, T ub)
{
  auto[_a, _b, _c, _f] = initEllipticODEwithK<N>(f, a, b, ua, ub);

  return Matrix<double, N - 1, N - 1>::triDiagThomas(_a, _b, _c, _f);
}
{% endhighlight %}

## Example
{% highlight c++ %}
double exact(double x)
{
  return 2.5 + 2.5 * x - 1 / (M_PI * M_PI) * std::sin(M_PI * x);
}

int main()
{
  constexpr int N = 10;
  auto a = 0.0, b = 1.0, ua = 2.5, ub = 5.0;
  auto h = (b - a) / N;

  auto calcVal =
    solveEllipticODEwithK<N>([](double x) { return std::sin(M_PI * x); }, a, b, ua, ub);


  std::array<double, N - 1> realVal;
  for (auto i = 1u; i < N; ++i)
  {
    realVal[i - 1] = exact(a + i * h);
  }

  std::cout << "Calculated Values\n" << calcVal << std::endl;
  std::cout << "Real Values\n" << realVal << std::endl;

  std::cout << "P1 norm:  " << pNorm(realVal - calcVal, 1) << std::endl;
  std::cout << "P2 norm:  " << pNorm(realVal - calcVal, 2) << std::endl;
  std::cout << "inf norm: " << infNorm(realVal - calcVal) << std::endl;

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
 Calculated Values
[     2.7184      2.94    3.1674    3.4028    3.6478    3.9028    4.1674      4.44    4.7184 ]

 Real Values
[     2.7187    2.9404     3.168    3.4036    3.6487    3.9036     4.168    4.4404    4.7187 ]

P1 norm:  0.0052875
P2 norm:  0.0018726
inf norm: 0.00083746

```

**Last Modification date:** 08 February 2018
