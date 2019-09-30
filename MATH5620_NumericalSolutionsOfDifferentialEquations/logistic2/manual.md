---
title: LogisticSolver
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Logistic Differential Equation

**Routine Name:** logisticSolver

**Author:** Philip Nelson

**Language:** C++

## Description

`logisticSolver` generates a function that takes time and returns the solution to the exact solution to the logistic differential equation:

\\[ \frac{dP}{dt} = \alpha P + \beta P^2 \\]

The exact solution has the form:

\\[\frac{\alpha}{(\frac{\alpha}{P_0}-\Beta)e^{-\alpha t}+\Beta}\\]

## Input

`logisticSolver(T const& a, T const& b, T const& p0)` requires:
* `a` - \\(\alpha\\)
* `b` - \\(\beta\\)
* `p0` - initial condition \\(P(0)\\)

## Output

Logistic returns a function that takes a \\(t\\) and evaluates the logistic equation at that time.

## Code
{% highlight c++ %}
template <typename T>
auto logisticSolver(T const& a, T const& b, T const& p0)
{
  return [=](T const& t) {
    return (a / (((a - p0 * b) / p0) * std::exp(-a * t) + b));
  };
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  double a = 2.5;
  double b = 1.7;
  double p0 = 3.2;

  std::cout << "alpha:\t" << a << "\nbeta:\t" << b << "\nP0:\t" << p0 << "\n\n";

   auto solveLog = logisticSolver(a, b, p0);

  // Call it for some basic values
  for (int t = -10; t < 0; ++t) {
    std::cout << "t = " << t << " -> " << solveLog(t)
      << "\tt = " << t+10 << " -> " << solveLog(t+10) << '\n';
  }

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
alpha:	2.5
beta:	1.7
P0:	3.2

t = -10 -> -3.77903e-11	t = 0 -> 3.2
t = -9 -> -4.6038e-10	t = 1 -> 1.53886
t = -8 -> -5.60858e-09	t = 2 -> 1.47596
t = -7 -> -6.83265e-08	t = 3 -> 1.47103
t = -6 -> -8.32388e-07	t = 4 -> 1.47062
t = -5 -> -1.01406e-05	t = 5 -> 1.47059
t = -4 -> -0.000123548	t = 6 -> 1.47059
t = -3 -> -0.00150653	t = 7 -> 1.47059
t = -2 -> -0.018566	t = 8 -> 1.47059
t = -1 -> -0.263361	t = 9 -> 1.47059
```

**Last Modification date:** 3 April 2018
