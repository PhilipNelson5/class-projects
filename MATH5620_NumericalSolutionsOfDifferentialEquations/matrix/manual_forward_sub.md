---
title: Forward Substitution
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Forward Substitution

**Routine Name:** forwardSub

**Author:** Philip Nelson

**Language:** C++

## Description

`forwardSub` solves a linear system \\(Ly=b\\) for \\(y\\) by forward substitution where \\(L\\) is a lower triangular matrix.

## Input

`forwardSub(std::array<T, M> b)` is called by a `Matrix<T, M, M> of type `T` and size `MxM` and requires:

* `std::array<T, M> b` - a column vector `b` of type `T` and size `M`

## Output

`forwardSub` returns a `std::array<T,M>` with the solution vector \\(y\\)

## Code
{% highlight c++ %}
std::array<T, M> forwardSub(std::array<T, M> b)
{
  std::array<T, M> y;
  for (auto i = 0u; i < N; ++i)
  {
    T sum = 0.0;
    for (auto j = 0u; j < i; ++j)
    {
      sum += m[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }
  return y;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 4, 4> L({ { 1,  0,  0,  0},
                           { 5,  1,  0,  0},
                           { 4, -6,  1,  0},
                           {-4,  5, -9,  1} });

  std::array<double, 4> y{3, 5, -6, 8};
  auto b = L * y;

  std::cout << " L\n" << L << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Real y\n" << y << std::endl;
  std::cout << " Calculated y\n";
  std::cout << L.forwardSub(b) << std::endl;
}
{% endhighlight %}

## Result
```
 L
|         1        0        0        0 |
|         5        1        0        0 |
|         4       -6        1        0 |
|        -4        5       -9        1 |

 b
[         3       20      -24       75 ]

 Real y
[         3        5       -6        8 ]

 Calculated y
[         3        5       -6        8 ]


```

**Last Modification date:** 07 February 2018
