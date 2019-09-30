---
title: Parallel Jacobi Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel Jacobi Iteration Software Manual

**Routine Name:** parallel_jacobi_iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./jacobiIteration.out** that can be executed.

**Description/Purpose:** The Jacobi method is an iterative algorithm for determining the solutions of a diagonally dominant system of linear equations. Each diagonal element is solved for, and an approximate value is plugged in. The process is then iterated until it converges. 

This code uses OpemMP to parallelize Jacobi Iteration.

**Input:** The routine takes a matrix, A, and a right hand side of the equation, b.

**Output:** The routine returns the solution, x, of the equation Ax = b.

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(4u);
  auto b = generate_right_side(A);
  auto x = parallel_jacobi_iteration(A, b);
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
|      -9.85      0.22    -0.776     -8.74 |
|       0.22        11      4.96     -2.34 |
|     -0.776      4.96      12.2      2.32 |
|      -8.74     -2.34      2.32     -17.8 |

 x
[          1         1         1         1 ]

 b
[      -19.1      13.8      18.7     -26.6 ]

 A * x
[      -19.1      13.8      18.7     -26.6 ]
```

_explanation of output_:

First, the matrix A is generated and displayed. It is a square matrix with uniformly distributed numbers and is symmetric and diagonally dominant. Then the rhs is computed and x is solved for and displayed. Finally b is shown and A * x is shown. We can see that b == A * x which is good.

**Implementation/Code:** The following is the code for parallel_jacobi_iteration

In this code, maceps returns a [std::tuple](https://en.cppreference.com/w/cpp/utility/tuple) with the machine epsilon and the precision. [std::get](https://en.cppreference.com/w/cpp/utility/tuple/get) is used to extract only the first value, the machine epsilon, from the returned tuple. The code also uses [std::fill](https://en.cppreference.com/w/cpp/algorithm/fill) to reset the x_new to all zeros each iteration.

``` cpp
template <typename T>
std::vector<T> parallel_jacobi_iteration(
  Matrix<T> A,
  std::vector<T> const& b,
  unsigned int const& MAX_ITERATIONS = 1000u)
{
  std::vector<T> x_new(b.size(), 0);
  std::vector<T> x(b.size(), 0);
  static const T macepsT = std::get<1>(maceps<T>());

  for (auto n = 0u; n < MAX_ITERATIONS; ++n)
  {
    std::fill(std::begin(x_new), std::end(x_new), 0);

#pragma omp parallel
    {
#pragma omp for
      for (auto i = 0u; i < A.size(); ++i)
      {
        T sum = 0.0;
        for (auto j = 0u; j < A.size(); ++j)
        {
          if (j == i) continue;
          sum += A[i][j] * x[j];
        }
        x_new[i] = (b[i] - sum) / A[i][i];
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
