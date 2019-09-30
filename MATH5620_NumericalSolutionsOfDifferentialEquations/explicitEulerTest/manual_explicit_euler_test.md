---
title: Explicit Euler Tests
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Explicit Euler Method Tests

**Author:** Philip Nelson

**Language:** C++

## Description

This tests the Explicit Euler method against the test cases in problem 1 of this assignment

## Code
{% highlight c++ %}
int main()
{
  double dt = 0.00001;

  double alpha = 10.0;
  std::vector lambdas = { 1.0, -1.0, 100.0 };

  double gamma = 0.1;
  double beta = 0.0001;
  std::vector Pos = { 25.0, 40000.0 };

  std::cout << "----- Lambda Differential Equation -----" << std::endl;
  for (const auto lambda : lambdas) {
    std::cout << explicit_euler( alpha, beta, gamma, dt,
        [=](double a, double b) {
        (void)b;
          return lambda * a;
        }
    ) << '\n';

    auto solveIVP = firstOrderIVPSolver(lambda, alpha);

    std::cout << solveIVP(dt);
  }

  std::cout << std::endl;
  std::cout << "----- Logistic Differential Equation -----" << std::endl;
  for (const auto p0 : Pos) {
    std::cout << explicit_euler( alpha, beta, gamma, dt,
        [=](double a, double b) {
        (void)b;
        return gamma * a - beta * a * a;
        }
    ) << '\n';

    std::cout << logistic(alpha, beta, dt, p0) << '\n';
  }

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
----- Lambda Differential Equation -----
exact	 t = 0 -> 10
approx	 t = 0 -> 10
exact	 t = 0.2 -> 12.214
approx	 t = 0.2 -> 12.214
exact	 t = 0.4 -> 14.9182
approx	 t = 0.4 -> 14.9182
exact	 t = 0.6 -> 18.2212
approx	 t = 0.6 -> 18.2211
exact	 t = 0.8 -> 22.2554
approx	 t = 0.8 -> 22.2553
exact	 t = 1 -> 27.1828
approx	 t = 1 -> 27.1824

lambda = -1
exact	 t = 0 -> 10
approx	 t = 0 -> 10
exact	 t = 0.2 -> 8.18731
approx	 t = 0.2 -> 8.1873
exact	 t = 0.4 -> 6.7032
approx	 t = 0.4 -> 6.70319
exact	 t = 0.6 -> 5.48812
approx	 t = 0.6 -> 5.4881
exact	 t = 0.8 -> 4.49329
approx	 t = 0.8 -> 4.49327
exact	 t = 1 -> 3.67879
approx	 t = 1 -> 3.67881

lambda = 100
exact	 t = 0 -> 10
approx	 t = 0 -> 10
exact	 t = 0.2 -> 4.85165e+09
approx	 t = 0.2 -> 4.80341e+09
exact	 t = 0.4 -> 2.35385e+18
approx	 t = 0.4 -> 2.30727e+18
exact	 t = 0.6 -> 1.14201e+27
approx	 t = 0.6 -> 1.10828e+27
exact	 t = 0.8 -> 5.54062e+35
approx	 t = 0.8 -> 5.32351e+35
exact	 t = 1 -> 2.68812e+44
approx	 t = 1 -> 2.55455e+44

----- Logistic Differential Equation -----

p0 = 25
exact	 t = 0 -> 25
approx	 t = 0 -> 25
exact	 t = 0.2 -> 25.4922
approx	 t = 0.2 -> 25.4922
exact	 t = 0.4 -> 25.9937
approx	 t = 0.4 -> 25.9937
exact	 t = 0.6 -> 26.5049
approx	 t = 0.6 -> 26.5049
exact	 t = 0.8 -> 27.0259
approx	 t = 0.8 -> 27.0259
exact	 t = 1 -> 27.5568
approx	 t = 1 -> 27.5568

p0 = 40000
exact	 t = 0 -> 40000
approx	 t = 0 -> 40000
exact	 t = 0.2 -> 22570.2
approx	 t = 0.2 -> 22569.9
exact	 t = 0.4 -> 15815.2
approx	 t = 0.4 -> 15815
exact	 t = 0.6 -> 12228
approx	 t = 0.6 -> 12227.8
exact	 t = 0.8 -> 10003.8
approx	 t = 0.8 -> 10003.7
exact	 t = 1 -> 8490.15
approx	 t = 1 -> 8490.11
```

**Last Modification date:** 3 April 2018
