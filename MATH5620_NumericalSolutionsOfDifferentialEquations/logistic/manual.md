---
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/class-projects/MATH5620_NumericalSolutionsOfDifferentialEquations/SoftwareManual"> Table of Contents </a>
# Logistic Differential Equaiton

**Routine Name:** logistic

**Author:** Philip Nelson

**Language:** C++

## Description

`logistic` returns the solution to the logistic differential equation given alpha, beta, time and an initial P. A make file is included with a driver program.

\\[ \frac{dP}{dt} = \alpha P + \beta P^2 \\]

```
$ make
$ ./logistic.out
```

This will compile and run the driver program.

## Input

`logistic( double a, double b, double t, double p0 )` requires:
* `a` - alpha
* `b` - beta
* `t` - time
* `p0` - initial condition \\(P(0)\\)

## Output

Logistic returns an `N`, which is the type of the initial parameter, with the solution to the logistic differential equation.

## Code
{% highlight c++ %}
template <typename A, typename B, typename T, typename N>
inline N logistic(A a, B b, T t, N p0)
{
  return a / (((a-p0*b)/p0) * exp(-a * t) + b);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  double a = 1.0;
  double b = 2.0;
  double t = 50;
  double p0 = 10.0;

  std::cout << "alpha:\t" << a << "\nbeta:\t" << b << "\ntime:\t" << t << "\nP0:\t" << p0 << std::endl;

  std::cout << "------------\nresult:\t" << logistic(a, b, t, p0) << std::endl;
}
{% endhighlight %}

## Result
```
alpha:	1
beta:	2
time:	50
P0:	10
-----------
result:	0.5
```

**Last Modification date:** 17 January 2018
