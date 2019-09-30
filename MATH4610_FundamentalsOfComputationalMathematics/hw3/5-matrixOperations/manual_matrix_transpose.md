---
title: Matrix Transpose
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Transpose Software Manual

**Routine Name:** transpose

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** This code will produce the transpose of any `m` x `n` matrix

**Input:** The code takes a single matrix as input

```
@tparam T Type of the elements in the matrix
@param m  The matrix
```

**Output:** The transpose of the input matrix

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m1 = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  Matrix<int> m2 = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
  };
  std::cout << "m1\n";
  std::cout << m1 << std::endl;
  std::cout << "transpose m1\n";
  std::cout << transpose(m1) << std::endl;
  std::cout << "m2\n";
  std::cout << m2 << std::endl;
  std::cout << "transpose m2\n";
  std::cout << transpose(m2) << std::endl;
}
```

**Output** from the lines above
```
m1
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

transpose m1
|          1         4         7 |
|          2         5         8 |
|          3         6         9 |

m2
|          1         2         3         4 |
|          5         6         7         8 |
|          9        10        11        12 |

transpose m2
|          1         5         9 |
|          2         6        10 |
|          3         7        11 |
|          4         8        12 |
```

_explanation of output_:

The matrix is printed then it's transpose. This is done for two matricies, `m1` and `m2`

**Implementation/Code:** The following is the code for

This code is generic for a matrix of any dimensionality and any type;

``` cpp
template <typename T>
Matrix<T> transpose(Matrix<T> const& m)
{
  // initialize the result matrix
  Matrix<T> tp(m[0].size());
  std::for_each(
    begin(tp), end(tp), [&](std::vector<T>& row) { row.resize(m.size()); });

  // for every column of m
  for (auto j = 0u; j < m[0].size(); ++j)
  {
    // for every row of m
    for (auto i = 0u; i < m.size(); ++i)
    {
      tp[j][i] = m[i][j];
    }
  }

  // return the transpose
  return tp;
}
```

**Last Modified:** October 2018
