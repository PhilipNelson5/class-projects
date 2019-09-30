---
title: Back Substitution
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Back Substitution Software Manual

**Routine Name:** back_substitution

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./backSub.out** that can be executed.

**Description/Purpose:** `backSub` solves a linear system \\(Ux=b\\) for \\(x\\) by back substitution where \\(U\\) is an upper triangular matrix.

**Input:**  takes two parameters, an upper triangular matrix and a vector:

```
@tparam T The type of the elements in U and b
@param U  A lower triangular matrix
@param b  A vector
```

**Output:** The routine returns the solution, x, to \\(Ux=b\\)

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double> U = {
    {3, 5, -6, 4},
    {0, 4, -6, 9},
    {0, 0, 3, 11},
    {0, 0, 0, -9}
  };

  std::vector<double> x = {4, 6, -7, 9};
  auto b = U * x;

  std::cout << " U\n" << U << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual x\n" << x << std::endl;
  std::cout << " Calculated x\n";
  std::cout << back_substitution(U, b) << std::endl;
}
```

**Output** from the lines above

```
 U
|          3         5        -6         4 |
|          0         4        -6         9 |
|          0         0         3        11 |
|          0         0         0        -9 |

 b
[        120       147        78       -81 ]

 Actual x
[          4         6        -7         9 ]

 Calculated x
[          4         6        -7         9 ]
```

_explanation of output_:

The matrix U is shown, then the constructed vector b = U * x. Next the x vector that was used to construct x, and finally the calculated x vector. We can see that the actual and calculated x vectors match.

**Implementation/Code:** The following is the code for back_substitution

``` cpp
template <typename T>
std::vector<T> back_substitution(Matrix<T> U, std::vector<T> b)
{
  std::vector<T> x(b.size());

  for (auto i = (int)U[0].size() - 1; i >= 0; --i)
  {
    T sum = 0.0;
    for (auto j = (unsigned)i + 1; j < U[0].size(); ++j)
    {
      sum += U[i][j] * x[j];
    }
    x[i] = (b[i] - sum) / U[i][i];
  }

  return x;
}
```

**Last Modification date:** October 2018
