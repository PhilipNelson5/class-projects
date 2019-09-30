---
title: Infinity Norm
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Infinity Norm

**Routine Name:** infNorm

**Author:** Philip Nelson

**Language:** C++

## Description

`infNorm` calculates the infinity norm of a vector which is also the maximum element

## Input

`infNorm(std::array<T, N> v)` requires:

* `std::array<T, N> v` - a column vector of type `T` and size `M`

## Output

A double with the value of the infinity norm

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
double infNorm(std::array<T, N> v)
{
  T max = std::abs(v[0]);
  for (auto&& x : v)
    max = std::max(max, std::abs(x));
  return max;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::array<double, 4> v = {11, 18, -20, 5};

  std::cout << " v\n" << v << std::endl;
  std::cout << "Infinity Norm: " << infNorm(v) << std::endl;
}
{% endhighlight %}

## Result
```
 v
[        11       18      -20        5 ]

Infinity Norm: 20
```

**Last Modification date:** 07 February 2018
