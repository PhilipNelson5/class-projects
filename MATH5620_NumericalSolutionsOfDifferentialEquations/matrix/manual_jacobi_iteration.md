---
title: Jacobi Iteration
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Jacobi Iteration

**Routine Name:** jacobiIteration

**Author:** Philip Nelson

**Language:** C++

## Description

`jacobiIteration` solves a linear system of equations by Jacobi Iteration.

## Input

`jacobiIteration(std::array<T, M> const& b, unsigned int const& MAX)` is called by a `Matrix<T, M, M> of type `T` and size `M`x`M` and requires:

* `std::array<T, M> b` - a column vector `b` of type `T` and size `M`
* `unsigned int MAX` - the max iterations (optional)

## Output

`jacobiIteration` returns a `std::array<T,M>` with the solution vector `x`

## Code
{% highlight c++ %}
std::array<T, M> jacobiIteration(std::array<T, M> const& b, unsigned int const& MAX = 1000)
{
  std::array<T, M> zeros;
  zeros.fill(0);

  std::array<T, M> x = zeros;

  for (auto n = 0u; n < MAX; ++n)
  {
    auto x_n = zeros;

    for (auto i = 0u; i < M; ++i)
    {
      T sum = 0;
      for (auto j = 0u; j < N; ++j)
      {
        if (j == i)
          continue;
        sum += m[i][j] * x[j];
      }
      x_n[i] = (b[i] - sum) / m[i][i];
    }

    if (allclose(x, x_n, maceps<T>().maceps))
      break;

    x = x_n;
  }

  return x;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 4, 4> A({
      {-2, 1, 0, 0},
      {1, -2, 1, 0},
      {0, 1, -2, 1},
      {0, 0, 1, -2}
      });
  std::array<double, 4> x = {4, 7, 2, 5};
  auto b = A*x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << "Calculated x\n";
  std::cout << A.jacobiIteration(b) << std::endl;
}
{% endhighlight %}

## Result
```
 A
|        -2        1        0        0 |
|         1       -2        1        0 |
|         0        1       -2        1 |
|         0        0        1       -2 |

 b
[        -1       -8        8       -8 ]

 x
[         4        7        2        5 ]

Calculated x
[         4        7        2        5 ]

```

**Last Modification date:** 07 February 2018
