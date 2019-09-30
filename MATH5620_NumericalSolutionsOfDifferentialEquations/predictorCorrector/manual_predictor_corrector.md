---
title: Predictor Corrector
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Predictor Corrector

**Routine Name:** `predictor_corrector`

**Author:** Philip Nelson

**Language:** C++

## Description

`predictor_corrector` belong to a class of algorithms designed to integrate ordinary differential equations – to find an unknown function that satisfies a given differential equation. All such algorithms proceed in two steps:

1. The initial, "prediction" step, starts from a function fitted to the function-values and derivative-values at a preceding set of points to extrapolate ("anticipate") this function's value at a subsequent, new point.

2. The next, "corrector" step refines the initial approximation by using the predicted value of the function and another method to interpolate that unknown function's value at the same subsequent point.
[1](https://en.wikipedia.org/wiki/Predictor–corrector_method)

Three families of linear multistep methods are commonly used: Adams–Bashforth methods, Adams–Moulton methods, and the backward differentiation formulas (BDFs).

The Adams–Bashforth methods are explicit methods. The coefficients are \\(a_{s-1}=-1\\) and \\(a_{s-2}=\cdots =a_0=0\\), while the \\(b_j\\) are chosen such that the methods has order s (this determines the methods uniquely).

The Adams–Moulton methods are similar to the Adams–Bashforth methods in that they also have \\(a_{s-1}=-1\\) and \\(a_{s-2}=\cdots =a_0=0\\). Again the b coefficients are chosen to obtain the highest order possible. However, the Adams–Moulton methods are implicit methods. By removing the restriction that \\(b_s = 0\\) , an s-step Adams–Moulton method can reach order s+1, while an s-step Adams–Bashforth methods has only order s.
[2](https://en.wikipedia.org/wiki/Linear_multistep_method)

## Input

`predictor_corrector(T x0, T y0, T x, T dt, F f)` requires:

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
T predictor_corrector(T x0, T y0, T x, T dt, F f, unsigned int n, const unsigned int MAX_ITERATIONS = 1000)
{
  double A, B, ALPHA, H, t, W[MAX_ITERATIONS], K1, K2, K3, K4;

  A = 0.0;
  B = 2.0;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(10);

  H = (B - A) / n;
  t = A;
  W[0] = 0.5; // initial value

  for (auto i = 1u; i <= 3u; i++)
  {
    K1 = H * f(t, W[i - 1]);
    K2 = H * f(t + H / 2.0, W[i - 1] + K1 / 2.0);
    K3 = H * f(t + H / 2.0, W[i - 1] + K2 / 2.0);
    K4 = H * f(t + H, W[i - 1] + K3);

    W[i] = W[i - 1] + 1 / 6.0 * (K1 + 2.0 * K2 + 2.0 * K3 + K4);

    t = A + i * H;

    std::cout << "At time " << t << " the solution = " << W[i] << std::endl;
  }

  for (auto i = 4u; i <= n; i++)
  {
    K1 = 55.0 * f(t, W[i - 1]) - 59.0 * f(t - H, W[i - 2]) +
         37.0 * f(t - 2.0 * H, W[i - 3]) - 9.0 * f(t - 3.0 * H, W[i - 4]);
    W[i] = W[i - 1] + H / 24.0 * K1;

    t = A + i * H;

    std::cout << "At time " << t << " the solution = " << W[i] << std::endl;
  }

  return W[i];
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout <<
    predictor_corrector(0.0, -1.0, 0.4, 0.1, [](double a, double b){return a*a+2*b;})
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
