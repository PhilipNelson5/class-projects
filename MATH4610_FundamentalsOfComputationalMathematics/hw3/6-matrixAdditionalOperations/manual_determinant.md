---
title: Determinant
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Determinant Software Manual

**Routine Name:** determinant

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts2.out** that can be executed.

**Description/Purpose:** The determinant is a value that can be computed from the elements of a square matrix. The determinant of a matrix A is denoted det(A), det A, or \|A\|. Geometrically, it can be viewed as the scaling factor of the linear transformation described by the matrix.

**Input:** A matrix

```
@tparam T The type of the elements in the matrix a
@param a  The matrix
```

**Output:** The determinant

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double> m1 = {
    { 1,  3,  2},
    {-3, -1, -3},
    { 2,  3,  1}
  };

  std::cout << m1 << '\n';

  std::cout << determinant(m1) << '\n';
}
```

**Output** from the lines above
```
m1
|          1         3         2 |
|         -3        -1        -3 |
|          2         3         1 |

determinant(m1)
-15
```

_explanation of output_:

first the matrix m1 is printed then the determinant of m1

**Implementation/Code:** The following is the code for determinant

Since the determinant is an inherently recursive operation, this function is implemented recursively. The base case is a 2x2 matrix which has a simple solution.

``` cpp
template <typename T>
T determinant(Matrix<T> const& a)
{
  // matrix must be square
  if (a.size() != a[0].size())
  {
    std::cerr << "ERROR: bad size in Determinant\n";
    exit(EXIT_FAILURE);
  }

  // base case, a 2x2 matrix
  if (a.size() == 2 && a[0].size() == 2)
  {
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
  }

  T det = 0;
  for (auto i = 0u; i < a.size(); ++i)
  {
    // find the determinant of the matrix removing row 0 and col i
    auto val = a[0][i] * determinant(removeRow(removeCol(a, i), 0));

    // subtract or add the value of
    // a[0][i] * the determinant of the sub-matrix
    if (i % 2)
      det -= val;
    else
      det += val;
  }

  return det;
}
```

**Last Modified:** October 2018
