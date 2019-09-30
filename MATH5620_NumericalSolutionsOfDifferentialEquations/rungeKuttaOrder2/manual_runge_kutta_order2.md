---
title: Runge Kutta Order 2
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Kunge Kutta Order 2

**Routine Name:** `runge_kutta_order2`

**Author:** Philip Nelson

**Language:** C++

## Description

`runge_kutta_order2` is in a family of implicit and explicit iterative methods, which include the well-known routine called the Euler Method, used in temporal discretization for the approximate solutions of ordinary differential equations. These methods were developed around 1900 by the German mathematicians C. Runge and M. W. Kutta.[1](https://en.wikipedia.org/wiki/Runge–Kutta_methods)

## Input

`runge_kutta_order2(F f, unsigned int n, T h, T x0, T y0)` requires:

* `F f` - the function
* `T x0` - the initial `x`
* `T y0` - the initial `y`
* `T h` - the step size

## Output

The value of `y` at `x`.

## Code
{% highlight c++ %}
template <typename T, typename F>
T runge_kutta_order2(F f, unsigned int n, T h, T x0, T y0)
{
  long double y[n], x[n], k[n][n];

  x[0] = x0;
  y[0] = y0;

  for (auto i = 1u; i <= n; i++)
  {
    x[i] = x[i - 1] + h;
  }

  for (auto j = 1u; j <= n; j++)
  {
    k[1][j] = h * f(x[j - 1], y[j - 1]);
    std::cout << "K[1] = " << k[1][j] << "\t";
    k[2][j] = h * f(x[j - 1] + h, y[j - 1] + k[1][j]);
    std::cout << "K[2] = " << k[2][j] << "\n";
    y[j] = y[j - 1] + ((k[1][j] + k[2][j]) / 2);
    std::cout << "y[" << h * j << "] = " << std::setprecision(5) << y[j]
              << std::endl;
  }
  return y;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout <<
    runge_kutta_order2(0.0, -1.0, 0.4, 0.1, [](double a, double b){return a*a+2*b;})
    << std::endl;
}
{% endhighlight %}

## Result
```
----- Lambda Differential Equation -----

lambda = 1
exact	0 -> 10
approx	0 -> 10
exact	0.2 -> 12.214
approx	0.2 -> 12.214
exact	0.4 -> 14.9182
approx	0.4 -> 14.9183
exact	0.6 -> 18.2212
approx	0.6 -> 18.2212

lambda = -1
exact	0 -> 10
approx	0 -> 10
exact	0.2 -> 8.18731
approx	0.2 -> 8.18732
exact	0.4 -> 6.7032
approx	0.4 -> 6.70321
exact	0.6 -> 5.48812
approx	0.6 -> 5.48813

lambda = 100
exact	0 -> 10
approx	0 -> 10
exact	0.2 -> 4.85165e+09
approx	0.2 -> 4.90044e+09
exact	0.4 -> 2.35385e+18
approx	0.4 -> inf
exact	0.6 -> 1.14201e+27

----- Logistic Differential Equaiton -----

p0 = 25
exact	0 -> 25
approx	0 -> 25
exact	0.2 -> 25.4922
approx	0.2 -> 25.4922
exact	0.4 -> 25.9937
approx	0.4 -> 25.9937
exact	0.6 -> 26.5049
approx	0.6 -> 26.5049

p0 = 40000
exact	0 -> 40000
approx	0 -> 40000
exact	0.2 -> 22570.2
approx	0.2 -> 22570.4
exact	0.4 -> 15815.2
approx	0.4 -> 15815.4
exact	0.6 -> 12228
approx	0.6 -> 12228.2
```

**Last Modification date:** 3 April 2018
