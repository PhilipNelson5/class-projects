---
title: Solve Linear System of Equations Gaussian Elimination
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Solve Linear System of Equation Gaussian Elimination Software Manual

**Routine Name:** solve_linear_system_gaussian_elimination

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./solveGauss.out** that can be executed.

**Description/Purpose:** This routine combines Gaussian elimination and back substitution to solve a linear system of equations \\(A x = b\\)

**Input:** The routine takes a matrix \\(A\\) and a vector \\(b\\) as inputs

**Output:** The routine returns the solution \\(x\\)

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

  std::vector<double> x = {4, 6, -7, 9};
  auto b = A * x;

  std::cout << " A\n" << A << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual x\n" << x << std::endl;

  std::cout << " Calculated x\n"
            << solve_linear_system_gaussian_elimination(m, b) << std::endl;
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
[         31        58        49        76 ]

 Actual x
[          4         6        -7         9 ]

 Calculated x
[          4         6        -7         9 ]
```

_explanation of output_:

The matrix A is shown, then the constructed vector b = U * x. Next the x vector that was used to construct b, and finally the calculated x vector. We can see that the actual and calculated x vectors match.

**Implementation/Code:** The following is the code for <++>

``` cpp
template <typename T>
std::vector<T> solve_linear_system_gaussian_elimination(Matrix<T>& m,
                                                        std::vector<T>& b)
{
  // create augmented matrix
  for (auto i = 0u; i < m.size(); ++i)
    m[i].push_back(b[i]);

  // perform Gaussian elimination of the augmented matrix
  gaussian_emlimination(m);

  // remove augmentation
  for (auto i = 0u; i < m.size(); ++i)
  {
    b[i] = m[i].back();
    m[i].pop_back();
  }

  // do back substitution
  auto x = back_substitution(m, b);

  // return the answer
  return x;
}
```

**Last Modified:** December 2018
