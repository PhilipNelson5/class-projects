---
title: Implicit Euler
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Implicit Euler Method

**Routine Name:** `implicit_euler`

**Author:** Philip Nelson

**Language:** C++

## Description

`implicit_euler`, also known as the backward Euler method, is one of the most basic numerical methods for the solution of ordinary differential equations. It is similar to the (standard) Euler method, but differs in that it is an implicit method. The backward Euler method has order one in time. [1](https://en.wikipedia.org/wiki/Backward_Euler_method)

## Code
{% highlight c++ %}
template <typename T, typename F>
implicit_euler (F f,T df,T x0,T t0,T tf,const unsigned int MAX_ITERATIONS)
{
  auto h=(tf-t0)/n;
  auto t=linspace(t0,tf,n+1);
  auto y=zeros(n+1,length(y0));
  auto y(1,:)=y0;
  for (auto i=0u; i < MAX_ITERATIONS; ++i)
  {
    x0=y(i,:);
    x1=x0-inv(eye(length(y0))-h*feval(df,t(i),x0))*(x0-h*feval(f,t(i),x0)’-y(i,:)’);
    x1 = newtons_method(f, dt, x0)
    y(i+1,:)=x1’;
  }
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout <<
    implicit_euler(0.0, -1.0, 0.4, 0.1, [](double a, double b){return a*a+2*b;})
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
