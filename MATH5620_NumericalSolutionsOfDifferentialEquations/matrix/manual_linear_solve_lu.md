---
title: Linear Solver LU
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Linear Solver by LU Factorization

**Routine Name:** solveLinearSystemLU

**Author:** Philip Nelson

**Language:** C++

## Description

`solveLinearSystemLU` solves a linear system of equations \\(Ax=b\\) by LU Factorization. The method used is:

\\[LU=PA\\]
\\[LUx = Pb\\]
\\[Ly = Pb\\]
\\[Ux=y\\]

## Input

`solveLinearSystemLU(std::array<T, M> b)` is called by a `Matrix<T, M, M>` of type `T` and size `M`x`M` and requires:

* `std::array<T, M> b` - a column vector `b` of type `T` and size `M`

## Output

`solveLinearSystemLU` returns a `std::array<T,M>` with the solution vector `x`

## Code
{% highlight c++ %}
std::array<T, M> solveLinearSystemLU(std::array<T, M> b)
{
  auto[L, U, P] = luFactorize();
  auto y = L.forwardSub(P * b);
  auto x = U.backSub(y);
  return x;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 4, 4> A(1, 10); // random 4x4 with values from 0-10
  std::array<double, 4> x = {4, 7, 2, 5};
  auto b = A*x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << "Calculated x\n";
  std::cout << A.solveLinearSystemLU(b) << std::endl;
}
{% endhighlight %}

## Result
```
 A
|         3        8        1        4 |
|         6       10        5        8 |
|         8        4        8        6 |
|         8       10        8        1 |

 b
[        90      144      106      123 ]

 Real x
[         4        7        2        5 ]

Calculated x
[         4        7        2        5 ]

```

**Last Modification date:** 07 February 2018
