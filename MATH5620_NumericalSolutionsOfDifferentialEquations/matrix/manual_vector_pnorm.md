---
title: P Norm
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# P Norm

**Routine Name:** pNorm

**Author:** Philip Nelson

**Language:** C++

## Description

`pNorm` calculates the p norm of a vector

## Input

`pNorm(std::array<T, N> v, unsigned int const& p)` requires:

* `std::array<T, N> v` - a column vector of type `T` and size `M`
* `unsigned int p` - the p norm that is desired

## Output

A double with the desired norm value.

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
double pNorm(std::array<T, N> v, unsigned int const& p)
{
  double sum = 0.0;
  for (auto&& x : v)
    sum += std::pow(std::abs(x), p);
  return std::pow(sum, 1.0 / p);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::array<double, 4> v = {11, 18, -20, 5};

  std::cout << " v\n" << v << std::endl;
  std::cout << "1 Norm: " << pNorm(v, 1) << std::endl;
  std::cout << "2 Norm: " << pNorm(v, 2) << std::endl;
}
{% endhighlight %}

## Result
```
 v
[        11       18      -20        5 ]

1 Norm: 54
2 Norm: 29.5
```

**Last Modification date:** 07 February 2018
