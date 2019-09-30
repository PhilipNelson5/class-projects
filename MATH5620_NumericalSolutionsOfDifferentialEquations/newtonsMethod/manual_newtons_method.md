---
title: Newton's Method
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Newton's Method

**Routine Name:** `newtons_method`

**Author:** Philip Nelson

**Language:** C++

## Description

`newtons_method`, In numerical analysis, Newton's method (also known as the Newton–Raphson method), named after Isaac Newton and Joseph Raphson, is a method for finding successively better approximations to the roots (or zeroes) of a real-valued function. It is one example of a root-finding algorithm. [1](https://en.wikipedia.org/wiki/Newton%27s_method)

## Input

`newtons_method(T x0, T y0, T x, T dt, F f)` requires:

* `F f` - the function
* `T dt` - the timestep
* `T x0` - the initial guess
* `const uint MAX_ITERATONS` - the number of iterations

## Output

The zero of the function at `x`.

## Code
{% highlight c++ %}
template <typename T, typename F>
T newtons_method(F f, T dt, T x0, const unsigned int MAX_ITERATONS = 100)
{
  auto tol = maceps<T>().maceps, i = 0u;
  while (std::abs(f(x0) - 0) > tol && ++i < MAX_ITERATONS)
  {
    x0 = x0 - f(x0) / ((f(x0 + dt) - f(x0)) / dt);
  }
  return x0;
}
{% endhighlight %}

**Last Modification date:** 3 April 2018
