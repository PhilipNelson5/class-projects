---
title: Gauss Sidel
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Gauss Sidel

**Routine Name:** gauss_sidel

**Author:** Philip Nelson

**Language:** C++

## Description

`gauss_sidel` is an iterative method used to solve a linear system of equations \\(Ax=b\\). It is named after the German mathematicians Carl Friedrich Gauss and Philipp Ludwig von Seidel, and is similar to the Jacobi method. Though it can be applied to any matrix with non-zero elements on the diagonals, convergence is only guaranteed if the matrix is either diagonally dominant, or symmetric and positive definite. [1](https://en.wikipedia.org/wiki/Gauss–Seidel_method)

## Input

```
gauss_sidel(Matrix<T, N, N>& A,
            std::array<T, N> const& b,
            unsigned int const& MAX_ITERATIONS = 1000u)
```
requires:

* `Matrix<T, N, N>& A` - an `N`x`N` matrix of type `T`
* `std::array<T, N> const& b` - the b vector
* `unsigned int const& MAX_ITERATIONS` - the maximum iterations (default 1000)

## Output

The solution vector \\(x\\).

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
std::array<T, N> gauss_sidel(Matrix<T, N, N>& A,
                             std::array<T, N> const& b,
                             unsigned int const& MAX_ITERATIONS = 1000u)
{
  std::array<T, N> x, xn;
  x.fill(0);
  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    xn.fill(0);
    std::cout << x << '\n';
    for (auto i = 0u; i < N; ++i)
    {
      auto s1 = 0.0, s2 = 0.0;
      for (auto j = 0u; j < i; ++j)
      {
        s1 += A[i][j] * xn[j];
      }
      for (auto j = i + 1; j < N; ++j)
      {
        s2 += A[i][j] * x[j];
      }
      xn[i] = (b[i] - s1 - s2) / A[i][i];
    }
    if (allclose(x, xn, maceps<T>().maceps))
    {
      return xn;
    }
    x = xn;
  }
  return x;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
int main()
{
  Matrix<double, 2, 2> A ({
      {16,   3},
      { 7, -11}
      });
  std::array<double, 2> b({
      {11, 13}
      });
  std::cout << "A\n" << A << "b\n" << b << '\n';
  std::cout << gauss_sidel(A, b, 100u) << "\n";
  std::cout << A.jacobiIteration(b, 100u) << "\n";

  Matrix<double, 4, 4> A1({
      {10, -1,  2,  0},
      {-1, 11, -1,  3},
      { 2, -1, 10, -1},
      { 0,  3, -1,  8}
      });
  std::array<double, 4> b1({
      {6, 25, -11, 15}
      });
  std::cout << "A\n" << A1 << "b\n" << b1 << '\n';
  std::cout << gauss_sidel(A1, b1, 100u) << "\n";
  std::cout << A1.jacobiIteration(b1, 100u) << "\n";
}
}
{% endhighlight %}

## Result
It is clear to see that as the matrix size increases, the Gauss Sidel method outperforms Jacobi Iteration.
```
A
|         16         3 |
|          7       -11 |
b
[         11        13 ]

Gauss Sidel completed in 18 iterations
[      0.812    -0.665 ]

Jacobi Iteration completed in 35 iterations
[      0.812    -0.665 ]

A
|         10        -1         2         0 |
|         -1        11        -1         3 |
|          2        -1        10        -1 |
|          0         3        -1         8 |
b
[          6        25       -11        15 ]

Gauss Sidel completed in 17 iterations
[          1         2        -1         1 ]

Jacobi Iteration completed in 43 iterations
[          1         2        -1         1 ]
```

**Last Modification date:** 20 March 2018
