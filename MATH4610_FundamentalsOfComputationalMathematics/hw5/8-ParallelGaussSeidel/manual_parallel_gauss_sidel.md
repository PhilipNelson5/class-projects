---
title: Parallel Gauss Seidel
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel Gauss Seidel Software Manual

**Routine Name:** parallel_gauss_seidel

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./parallelGaussSeidel.out** that can be executed.

**Description/Purpose:** Gauss Seidel is an iterative method used to solve a linear system of equations \\(Ax=b\\). It is named after the German mathematicians Carl Friedrich Gauss and Philipp Ludwig von Seidel, and is similar to the Jacobi method. Though it can be applied to any matrix with non-zero elements on the diagonals, convergence is only guaranteed if the matrix is either diagonally dominant, or symmetric and positive definite. [1](https://en.wikipedia.org/wiki/Gauss–Seidel_method)

This code uses OpemMP to parallelize Gauss Seidel. There will not be much speedup here because of the way that the method takes advantage of x_new within an iteration. This hurts parallelization because other threads need acces to the new x_new before they can proceed.

**Input:** The routine takes a matrix, A, and a right hand side, b.

**Output:** The routine returns x, the solution to A * x = b.

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(4u);
  auto b = generate_right_side(A);
  auto x = parallel_gauss_seidel(A, b);
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
|       16.4      6.47     0.579     -4.09 |
|       6.47      9.06       2.5     -5.42 |
|      0.579       2.5       6.5     0.139 |
|      -4.09     -5.42     0.139      -6.1 |

 x
[      0.772      2.51      1.44      2.54 ]

 b
[       19.4      12.6      9.72     -15.5 ]

 A * x
[       19.4      17.6      16.5       -32 ]
```

_explanation of output_:

First, the matrix A is generated and displayed. It is a square matrix with uniformly distributed numbers and is symmetric and diagonally dominant. Then the rhs is computed and x is solved for and displayed. Finally b is shown and A * x is shown. We can see that b == A * x which is good.

**Implementation/Code:** The following is the code for parallel_gauss_seidel

In this code, maceps returns a [std::tuple](https://en.cppreference.com/w/cpp/utility/tuple) with the machine epsilon and the precision. [std::get](https://en.cppreference.com/w/cpp/utility/tuple/get) is used to extract only the first value, the machine epsilon, from the returned tuple. The code also uses [std::fill](https://en.cppreference.com/w/cpp/algorithm/fill) to reset the x_new to all zeros each iteration.

``` cpp
template <typename T>
std::vector<T> gauss_seidel(Matrix<T>& A,
                            std::vector<T> const& b,
                            unsigned int const& MAX_ITERATIONS = 1000u)
{
  static const T macepsT = std::get<1>(maceps<T>());

  std::vector<T> x(b.size(), 0), x_new(b.size(), 0);

  for (auto k = 0u; k < MAX_ITERATIONS; ++k)
  {
    std::fill(std::begin(x_new), std::end(x_new), 0);
#pragma omp parallel
    {
#pragma omp for
      for (auto i = 0u; i < A.size(); ++i)
      {
        auto s1 = 0.0, s2 = 0.0;
        for (auto j = 0u; j < i; ++j)
        {
          s1 += A[i][j] * x_new[j];
        }
        for (auto j = i + 1; j < A.size(); ++j)
        {
          s2 += A[i][j] * x[j];
        }
        x_new[i] = (b[i] - s1 - s2) / A[i][i];
      }
    }

    if (allclose(x, x_new, macepsT))
    {
      return x_new;
    }
    x = x_new;
  }
  return x;
}
```

**Last Modified:** December 2018
