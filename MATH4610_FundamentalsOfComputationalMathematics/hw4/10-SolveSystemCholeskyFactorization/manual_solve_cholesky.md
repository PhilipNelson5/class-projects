---
title: Linear Solver Cholesky
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Linear Solver by LU Factorization Software Manual

**Routine Name:** solve_linear_system_cholesky

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./cholesky.out** that can be executed.


**Description/Purpose:**

Solve a linear system of equations \\(Ax = b\\) by using Cholesky decomposition.

**Input:** A square matrix A and a vector b

```
@tparam T The type of the elements of A and b
@param  A The matrix of linear systems of equations 
@param  b The right hand side
```

**Output:** The solutions in a vector

```
@return A vector of solutions x
```

**Usage/Example:**

``` cpp
int main()
{
  const auto n = 5;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;

  std::vector<double> x = {4, 7, 2, 5, 4};
  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << " Calculated x\n";
  std::cout << solve_linear_system_cholesky(A, b) << std::endl;
}
```

**Output** from the lines above
```
 A
|        200       -35       125      2.08      54.6 |
|        -35       153     -18.9      25.7      12.7 |
|        125     -18.9       123      6.21      31.2 |
|       2.08      25.7      6.21       119      99.7 |
|       54.6      12.7      31.2      99.7       266 |

 b
[   1.04e+03  1.07e+03       769   1.2e+03  1.93e+03 ]

 x
[          4         7         2         5         4 ]

 Calculated x
[          4         7         2         5         4 ]
```

_explanation of output_:

The matrix A is shown, then the constructed vector b = A * x. Next the x vector that was used to construct b, and finally the calculated x vector. We can see that the actual and calculated x vectors match.

**Implementation/Code:** The following is the code for solve_linear_system_cholesky

``` cpp
template <typename T>
std::vector<T> solve_linear_system_cholesky(Matrix<T> A, std::vector<T> b)
{
  auto L = cholesky_factorization(A);
  auto U = transpose(L);
  auto y = forward_substitution(L, b);
  auto x = back_substitution(U, y);
  return x;
}
```

**Last Modified:** December 2018
