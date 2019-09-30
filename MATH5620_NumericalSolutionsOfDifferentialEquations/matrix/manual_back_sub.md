---
title: Back Substitution
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Back Substitution

**Routine Name:** backSub

**Author:** Philip Nelson

**Language:** C++

## Description

`backSub` solves a linear system \\(Ux=b\\) for \\(x\\) by back substitution where \\(U\\) is an upper triangular matrix.

## Input

`backSub(std::array<T, M> b)` is called by a `Matrix<T, M, M> of type `T` and size `MxM` and requires:

* `std::array<T, M> b` - a column vector `b` of type `T` and size `M`

## Output

`backSub` returns a `std::array<T,M>` with the solution vector \\(x\\)

## Code
{% highlight c++ %}
std::array<T, M> backSub(std::array<T, M> b)
{
  std::array<T, M> x;
  for (auto i = (int)M - 1; i >= 0; --i)
  {
    T sum = 0.0;
    for (auto j = (unsigned int)i + 1; j < M; ++j)
    {
      sum += m[i][j] * x[j];
    }
    x[i] = (b[i] - sum) / m[i][i];
  }
  return x;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 4, 4> U({ { 3,  5, -6,  4},
                           { 0,  4, -6,  9},
                           { 0,  0,  3, 11},
                           { 0,  0,  0, -9} });

  std::array<double, 4> x{4, 6, -7, 9};
  auto b = U * x;

  std::cout << " U\n" << U << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Real x\n" << x << std::endl;
  std::cout << " Calculated x\n";
  std::cout << U.backSub(b) << std::endl;
}
{% endhighlight %}

## Result
```
 U
|         3        5       -6        4 |
|         0        4       -6        9 |
|         0        0        3       11 |
|         0        0        0       -9 |

 b
[       120      147       78      -81 ]

 Real x
[         4        6       -7        9 ]

 Calculated x
[         4        6       -7        9 ]
```

**Last Modification date:** 07 February 2018
