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

```
std::array<T, N> jacobiIteration(Matrix<T, N, N> A,
                                 std::array<T, N> const& b,
                                 unsigned int const& MAX = 1000u)
```
requires:

* `Matrix<T, N, N>& A` - an `N`x`N` matrix of type `T`
* `std::array<T, N> const& b` - the b vector
* `unsigned int const& MAX_ITERATIONS` - the maximum iterations (default 1000)

## Output

`jacobiIteration` returns a `std::array<T,M>` with the solution vector `x`

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
std::array<T, N> jacobiIteration(Matrix<T, N, N> A,
                                 std::array<T, N> const& b,
                                 unsigned int const& MAX = 1000u)
{
  auto ct = 0u;
  std::array<T, N> zeros;
  zeros.fill(0);

  std::array<T, N> x = zeros;

  for (auto n = 0u; n < MAX; ++n)
  {
    ++ct;
    auto x_n = zeros;

    for (auto i = 0u; i < N; ++i)
    {
      T sum = 0;
      for (auto j = 0u; j < N; ++j)
      {
        if (j == i) continue;
        sum += A[i][j] * x[j];
      }
      x_n[i] = (b[i] - sum) / A[i][i];
    }

    if (allclose(x, x_n, maceps<T>().maceps))
    {
      std::cout << "Jacobi Iteration completed in " << ct << " iterations\n";
      return x_n;
    }

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
      {-2,  1,  0,  0},
      { 1, -2,  1,  0},
      { 0,  1, -2,  1},
      { 0,  0,  1, -2}
      });
  std::array<double, 4> x = {
      {4, 7, 2, 5}
      };

  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;
  std::cout << jacobiIteration(A, b, 1000u) << std::endl;
}
{% endhighlight %}

## Result
```
 A
|         -2         1         0         0 |
|          1        -2         1         0 |
|          0         1        -2         1 |
|          0         0         1        -2 |

 b
[         -1        -8         8        -8 ]

 x
[          4         7         2         5 ]

Jacobi Iteration completed in 170 iterations
[          4         7         2         5 ]
```

**Last Modification date:** 07 February 2018
