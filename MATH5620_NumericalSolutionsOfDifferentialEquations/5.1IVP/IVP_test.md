---
title: IVP Test
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# First Order IVP Solver

**Routine Name:** firstOrderIVPSolver

**Author:** Philip Nelson

**Language:** C++

## Description

This method is used to solve the simple first order initial value problem

\\[u` = \lambda u\\]

with \\(u(0)=P_0\\). This ordinary differential equations has the soltion

\\[u(t)=\alpha e^{\lambda t}\\]

## Input

`firstOrderIVPSolver(const T& lambda, const T& alpha)` requires:

* `const T& l` - \\(\lambda\\)
* `const T& a` - \\(\alpha\\)

## Output

This method returns a function that can be evaluated at any time \\(t\\)
to obtain the exact solution to the analytic equation
(including roundoff error).

## Code
{% highlight c++ %}
template <typename T>
auto firstOrderIVPSolver(const T& l, const T& a)
{
  return [=](const T& t) { return a * std::exp(l * t); };
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{

  auto solveIVP = firstOrderIVPSolver(-1.5, 7.3);

  for(auto t = 0; t < 10; ++t)
  {
    std::cout << "t = " << t << " -> " << solveIVP(t)
      << "\tt = " << t+10 << " -> " << solveIVP(t+10) << '\n';
  }

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
t = 0 -> 7.3		t = 10 -> 2.23309e-06
t = 1 -> 1.62885	t = 11 -> 4.98269e-07
t = 2 -> 0.363446	t = 12 -> 1.11179e-07
t = 3 -> 0.0810957	t = 13 -> 2.48074e-08
t = 4 -> 0.0180949	t = 14 -> 5.53527e-09
t = 5 -> 0.00403752	t = 15 -> 1.23509e-09
t = 6 -> 0.000900892	t = 16 -> 2.75585e-10
t = 7 -> 0.000201016	t = 17 -> 6.14913e-11
t = 8 -> 4.48528e-05	t = 18 -> 1.37206e-11
t = 9 -> 1.0008e-05	t = 19 -> 3.06147e-12
```

**Last Modification date:** 3 April 2018
