---
title: Linear Solver LU
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Linear Solver by LU Factorization Software Manual

**Routine Name:** solve_linear_system_LU

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./luf.out** that can be executed.


**Description/Purpose:**

Solve a linear system of equations \\(Ax = b\\) by LU Factorization. The method used is:

\\[LU = PA\\]
\\[LUx = Pb\\]
\\[Ly = Pb\\]
\\[Ux = y\\]

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
  Matrix<double> A = {
    {1, 2, 3, 4},
    {4, 5, 6, 6},
    {2, 5, 1, 2},
    {7, 8, 9, 7}
  };
  std::vector<double> x = {4, 7, 2, 5};
  auto b = A * x;
  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " x\n" << x << std::endl;

  std::cout << " Calculated x\n";
  std::cout << solve_linear_system_LU(A, b) << std::endl;
}
```

**Output** from the lines above
```
 A
|          1         2         3         4 |
|          4         5         6         6 |
|          2         5         1         2 |
|          7         8         9         7 |

 b
[         44        93        55       137 ]

 x
[          4         7         2         5 ]

 Calculated x
[          4         7         2         5 ]

```

_explanation of output_:

The matrix A is shown, then the constructed vector b = U * x. Next the x vector that was used to construct b, and finally the calculated x vector. We can see that the actual and calculated x vectors match.

**Implementation/Code:** The following is the code for solve_linear_system_LU

``` cpp
template <typename T>
std::vector<T> solve_linear_system_LU(Matrix<T> A, std::vector<T> b)
{
  auto [L, U, P] = LU_factorize(A);
  auto y = forward_substitution(L, P * b);
  auto x = back_substitution(U, y);
  return x;
}
```

**Last Modified:** October 2018
