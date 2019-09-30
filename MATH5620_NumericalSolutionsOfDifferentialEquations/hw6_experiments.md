---
title: Homework 6
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Homework 6

**Language:** C++

## Description

Test problems from 7.1, 7.2, and 7.3 of the text.

{% highlight c++ %}
int main() {
    cout << "Problem 1: Explicit Euler" << endl;
    cout << Euler(0, 1, 2, .001, .001, 2000, [=](T y, T yk){ return y - yk - h * y; }) << endl;               // 7.1
    cout << Euler(0, 1, 2, .001, .001, 2000, [=](T y, T yk){ return y - yk + h * y; }) << endl;               // 7.2
    cout << Euler(0, 1, 2, .001, .001, 2000, [=](T y, T yk){ return y - yk + 100 * h * y; }) << endl << endl; // 7.3


    cout << "Problem 2: Implicit Euler" << endl;
    cout << impEuler(0, 1, 2, .001, .001, 2000,
      [=](T y, T yk){ return y-yk - h*y; }, [=](T y, T yk){ return 1 - h; }) << endl;                    // 7.1
    cout << impEuler(0, 1, 2, .001, .001, 2000,
      [=](T y, T yk){ return y- yk + h*y; }, [=](T y, T yk){ return 1 + h; }) << endl;                   // 7.2
    cout << impEuler(0, 1, 2, .001, .001, 2000,
      [=](T y, T yk){ return y- yk + 100*h*y; }, [=](T y, T yk){ return 1 + 100 * h; }) << endl << endl; // 7.3


    cout << "Problem 3: Explicit Euler 10^-2" << endl;
    cout << Euler(0, 1, 2, .01, .001, 2000, [=](T y, T yk){ return y - yk - h * y; }) << endl;               // 7.1
    cout << Euler(0, 1, 2, .01, .001, 2000, [=](T y, T yk){ return y - yk + h * y; }) << endl;               // 7.2
    cout << Euler(0, 1, 2, .01, .001, 2000, [=](T y, T yk){ return y - yk + 100 * h * y; }) << endl << endl; // 7.3


    cout << "Explicit 10^-1" << endl;
    cout << Euler(0, 1, 2, .1, .001, 2000, [=](T y, T yk){ return y - yk - h * y; }) << endl;               // 7.1
    cout << Euler(0, 1, 2, .1, .001, 2000, [=](T y, T yk){ return y - yk + h * y; }) << endl;               // 7.2
    cout << Euler(0, 1, 2, .1, .001, 2000, [=](T y, T yk){ return y - yk + 100 * h * y; }) << endl << endl; // 7.3


    cout << "Problem 4: Adam`s Bashforth - Adam's Moulton" << endl;
    cout << AdamBashMoul3(0, 1, 2, .001, [=](T y, T yk){ return y - yk - h * y; }) << endl;               // 7.1
    cout << AdamBashMoul3(0, 1, 2, .001, [=](T y, T yk){ return y - yk + h * y; }) << endl;               // 7.2
    cout << AdamBashMoul3(0, 1, 2, .001, [=](T y, T yk){ return y - yk + 100 * h * y; }) << endl << endl; // 7.3


    cout << "Problem 5: Runge Kutta" << endl;
    cout << RungeKutta4(0, 1, 2, .001, [=](T y, T yk){ return y - yk - h * y; }) << endl;               // 7.1
    cout << RungeKutta4(0, 1, 2, .001, [=](T y, T yk){ return y - yk + h * y; }) << endl;               // 7.2
    cout << RungeKutta4(0, 1, 2, .001, [=](T y, T yk){ return y - yk + 100 * h * y; }) << endl << endl; // 7.3
}
{% endhighlight %}

## Results:

```
Problem 1: Explicit Euler
-0.415692
-0.416163
-1.45252e+76


Problem 2: Implicit Euler
-0.416601
-0.420292
-0.833246


Problem 3: Explicit Euler 10^-2
-0.411589
-0.416309
-3.82603e+254


Explicit 10^-1
-0.369502
-0.41792
-6.01633e+41


Problem 4: Adam`s Bashforth - Adam's Moulton
-0.416147
-0.416147
7.69951e+283


Problem 5: Runge Kutta
-0.416146
-0.416147
-0.416147
```

**Last Modification date:** 3 April 2018
