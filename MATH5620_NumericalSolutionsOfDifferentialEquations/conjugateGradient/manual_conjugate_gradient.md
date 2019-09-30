---
title: Conjugate Gradient
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Conjugate Gradient Method

**Routine Name:** `conjugate_gradient`

**Author:** Philip Nelson

**Language:** C++

## Description

`conjugate_gradient` is an algorithm for the numerical solution of particular systems of linear equations, namely those whose matrix is symmetric and positive-definite. `conjugate_gradient` is implemented as an iterative algorithm, applicable to sparse systems that are too large to be handled by a direct implementation. [1](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

## Input


```
std::array<T, N> conjugate_gradient(Matrix<T, N, N>& A,
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
std::array<T, N> conjugate_gradient(Matrix<T, N, N>& A,
                                    std::array<T, N> const& b,
                                    unsigned int const& MAX_ITERATIONS = 1000u)
{
  auto ct = 0u;
  auto tol = maceps<T>().maceps;
  std::array<T, N> x_k, x_k1;
  x_k.fill(0);
  x_k1.fill(0);
  auto r_k = b;
  auto r_k1 = r_k, r_k2 = r_k;
  auto p_k = r_k, p_k1 = r_k;
  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    ++ct;
    if (k != 0)
    {
      auto b_k = (r_k1 * r_k1) / (r_k2 * r_k2);
      p_k = r_k1 + b_k * p_k1;
    }
    auto s_k = A * p_k;
    auto a_k = r_k1 * r_k1 / (p_k * s_k);
    x_k = x_k1 + a_k * p_k;
    r_k = r_k1 - a_k * s_k;

    if (allclose(x_k, x_k1, tol))
    {
      std::cout << "Conjugate Gradient completed in " << ct << " iterations\n";
      return x_k;
    }

    r_k2 = r_k1;
    r_k1 = r_k;
    x_k1 = x_k;
    p_k1 = p_k;
  }
  return x_k;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 2, 2> A({
      {16,   3},
      { 7, -11}
      });
  std::array<double, 2> b({
      {11, 13}
      });
  std::cout << "A\n" << A << "b\n" << b << '\n';
  std::cout << gauss_sidel(A, b, 100u) << "\n";
  std::cout << A.jacobiIteration(b, 100u) << "\n";
  std::cout << conjugate_gradient(A, b, 10000u) << "\n";

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
  std::cout << conjugate_gradient(A1, b1, 1000u) << "\n";
}
{% endhighlight %}

## Result
As opposed to Gauss Sidel, Conjugate Gradient converges differently on the solution depending on the topology of the region around the solution.
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

Conjugate Gradient completed in 75 iterations
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

Conjugate Gradient completed in 5 iterations
[          1         2        -1         1 ]
```

**Last Modification date:** 21 March 2018
