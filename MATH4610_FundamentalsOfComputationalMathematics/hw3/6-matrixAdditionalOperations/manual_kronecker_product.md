---
title: Kronecker Product
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Kronecker Product Software Manual

**Routine Name:** Kronecker Product

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts2.out** that can be executed.

**Description/Purpose:** In mathematics, the Kronecker product is an operation on two matrices of arbitrary size resulting in a block matrix. It is a generalization of the outer product from vectors to matrices, and gives the matrix of the tensor product with respect to a standard choice of basis.

**Input:** Two matrices m1 and m2

```
@tparam T The type of the elements stored in m1 and m2
@param m1 The first matrix
@param m2 The second matrix
```

**Output:** A matrix with the result of the Kronecker Product

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double>
  m1 = {
    {1, 2},
    {3, 4},
    {1, 0}
  },
  m2 = {
    {0, 5, 2},
    {6, 7, 3}
  };
  std::cout << m1 << '\n' << m2 << '\n';

  std::cout << kronecker_product(m1, m2) << '\n';
}
```

**Output** from the lines above
```
m1
|          1         2 |
|          3         4 |
|          1         0 |

m2
|          0         5         2 |
|          6         7         3 |

Krocecker Product of m1 and m2
|          0         5         2         0        10         4 |
|          6         7         3        12        14         6 |
|          0        15         6         0        20         8 |
|         18        21         9        24        28        12 |
|          0         5         2         0         0         0 |
|          6         7         3         0         0         0 |
```

_explanation of output_:

m1 and m2 are output first, then the result of the Kronecker Product is displayed.

**Implementation/Code:** The following is the code for kronecker_product

``` cpp
template <typename T>
Matrix<T> kronecker_product(Matrix<T> m1, Matrix<T> m2)
{
  // get the dimensionality of m1 and m2
  auto r1 = m1.size(), c1 = m1[0].size(), r2 = m2.size(), c2 = m2[0].size();

  // initialize the result matrix mr which is r1*r2 x c1*c2
  Matrix<T> mr(r1 * r2);
  std::for_each(std::begin(mr), std::end(mr), [&c1, c2](auto& row) {
    row.resize(c1 * c2);
  });

  // for each row in matrix 1
  for (auto i = 0u; i < r1; ++i)
  {
    // for each col in matrix 1
    for (auto j = 0u; j < c1; ++j)
    {
      // for each row in matrix 2
      for (auto k = 0u; k < r2; ++k)
      {
        // for each col in matrix 2
        for (auto l = 0u; l < c2; ++l)
        {
          // Each element of matrix m1 is
          // multiplied by the whole matrix m2
          // and stored in matrix mr
          mr[i * r2 + k][j * c2 + l] = m1[i][j] * m2[k][l];
        }
      }
    }
  }

  return mr;
}
```

**Last Modified:** October 2018
