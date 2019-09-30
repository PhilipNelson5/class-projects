---
title: Jacobi Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Jacobi Iteration Software Manual

**Routine Name:** jacobi_iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./jacobiIteration.out** that can be executed.

**Description/Purpose:** The Jacobi method is an iterative algorithm for determining the solutions of a diagonally dominant system of linear equations. Each diagonal element is solved for, and an approximate value is plugged in. The process is then iterated until it converges. 

**Input:** The routine takes a matrix, A, and a right hand side of the equation, b.

**Output:** The routine returns the solution, x, of the equation Ax = b.

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(4u);
  auto b = generate_right_side(A);
  auto x = jacobi_iteration(A, b);
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
|      -5.36     -3.94     -1.67     -1.03 |
|      -3.94     -13.7     -1.89     -9.48 |
|      -1.67     -1.89        14      7.94 |
|      -1.03     -9.48      7.94     -16.8 |

 x
[          1         1         1         1 ]

 b
[        -12     -29.1      18.4     -19.3 ]

 A * x
[        -12     -29.1      18.4     -19.3 ]
```

_explanation of output_:

First, the matrix A is generated and displayed. It is a square matrix with uniformly distributed numbers and is symmetric and diagonally dominant. Then the rhs is computed and x is solved for and displayed. Finally b is shown and A * x is shown. We can see that b == A * x which is good.

**Implementation/Code:** The following is the code for jacobi_iteration

In this code, maceps returns a [std::tuple](https://en.cppreference.com/w/cpp/utility/tuple) with the machine epsilon and the precision. [std::get](https://en.cppreference.com/w/cpp/utility/tuple/get) is used to extract only the first value, the machine epsilon, from the returned tuple. The code also uses [std::fill](https://en.cppreference.com/w/cpp/algorithm/fill) to reset the x_new to all zeros each iteration.

``` cpp
template <typename T>
std::vector<T> jacobi_iteration(Matrix<T> A,
                                std::vector<T> const& b,
                                unsigned int const& MAX_ITERATIONS = 1000u)
{
  std::vector<T> zeros(b.size(), 0);
  std::vector<T> x(b.size(), 0);
  static const T macepsT = std::get<1>(maceps<T>());

  for (auto n = 0u; n < MAX_ITERATIONS; ++n)
  {
    auto x_n = zeros;

    for (auto i = 0u; i < A.size(); ++i)
    {
      T sum = 0.0;
      for (auto j = 0u; j < A.size(); ++j)
      {
        if (j == i) continue;
        sum += A[i][j] * x[j];
      }
      x_n[i] = (b[i] - sum) / A[i][i];
    }

    if (allclose(x, x_n, macepsT))
    {
      return x_n;
    }

    x = x_n;
  }

  return x;
}
```

**Last Modified:** December 2018
