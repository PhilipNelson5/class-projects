---
title: Explicit Euler
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Explicit Euler Method

**Routine Name:** `explicit_euler`

**Author:** Philip Nelson

**Language:** C++

## Description

`explicit_euler` is a first-order numerical procedure for solving ordinary differential equations (ODEs) with a given initial value. It is the most basic explicit method for numerical integration of ordinary differential equations and is the simplest Runge–Kutta method. [1](https://en.wikipedia.org/wiki/Euler_method)

## Input


`explicit_euler(T x0, T y0, T x, T dt, F f)` requires:

* `T x0` - the initial `x`
* `T y0` - the initial `y`
* `T x` - the value of `x` for which you want to find value of `y`
* `T dt` - the delta t step
* `F f` - the function defining \\(\frac{dy}{dx}\\)

## Output

The value of `y` at `x`.

## Code
{% highlight c++ %}
template <typename T, typename F>
T explicit_euler(T x0, T y0, T x, T dt, F f)
{
  auto tol = maceps<T>().maceps;
  while (std::abs(x - x0) > tol)
  {
    y0 = y0 + (dt * f(x0, y0));
    x0 += dt;
  }
  return y0;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout <<
    explicit_euler(0.0, -1.0, 0.4, 0.1, [](double a, double b){return a*a+2*b;})
    << std::endl;
}
{% endhighlight %}

## Result
```
-2.05836
```

**Last Modification date:** 21 March 2018
