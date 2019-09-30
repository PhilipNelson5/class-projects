---
title: Parallel Conjugate Gradient
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel Contents Gradient Software Manual

**Routine Name:** parallel_contents_gradient

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./conjugateGradient.out** that can be executed.

**Description/Purpose:** Conjugate Gradient is an algorithm for the numerical solution of particular systems of linear equations, namely those whose matrix is symmetric and positive-definite. `conjugate_gradient` is implemented as an iterative algorithm, applicable to sparse systems that are too large to be handled by a direct implementation. [1](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

This code uses OpenMP to parallelize the iterations

**Input:** This routine takes two inputs, a matrix A, and a right hand side b

**Output:** The routine returns the solution x to the equation A * x = b.

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(4u);
  auto b = generate_right_side(A);
  auto x = parallel_conjugate_gradient(A, b);
  auto Ax = A * x;

  std::cout << " A\n" << A << std::endl;
  std::cout << " x\n" << x << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " A * x\n" << Ax << std::endl;
}
```

**Output** from the lines above
```
 A
|         10        -1         2         0 |
|         -1        11        -1         3 |
|          2        -1        10        -1 |
|          0         3        -1         8 |

 x
[          1         2        -1         1 ]

 b
[          6        25       -11        15 ]

 A * x
[          6        25       -11        15 ]
```

_explanation of output_:

First, the matrix A is generated and displayed. It is a square matrix with uniformly distributed numbers and is symmetric and diagonally dominant. Then the rhs is computed and x is solved for and displayed. Finally b is shown and A * x is shown. We can see that b == A * x which is good.

**Implementation/Code:** The following is the code for parallel_conjugate_gradient

``` cpp
template <typename T>
std::vector<T> parallel_conjugate_gradient(
  Matrix<T>& A,
  std::vector<T> const& b,
  unsigned int const& MAX_ITERATIONS = 1000u)
{
  static const T tol = std::get<1>(maceps<T>());
  std::vector<T> x_k(A.size(), 0), x_k1(A.size(), 0);
  std::vector<T> r_k = b;
  std::vector<T> r_k1 = r_k, r_k2 = r_k;
  std::vector<T> p_k = r_k, p_k1 = r_k;

#pragma omp parallel
  {
#pragma omp for
    for (auto k = 0u; k < MAX_ITERATIONS; ++k)
    {
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
        return x_k;
      }

      r_k2 = r_k1;
      r_k1 = r_k;
      x_k1 = x_k;
      p_k1 = p_k;
    }
  }

  return x_k;
}
```

**Last Modified:** December 2018
