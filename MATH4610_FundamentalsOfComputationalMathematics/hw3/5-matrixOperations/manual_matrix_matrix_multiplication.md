---
title: Matrix Matrix Multiplication
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Matrix Multiplication Software Manual

**Routine Name:** Matrix Matrix Multiplication

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** This routine overloads the `*` operator to multiply a matrix by a matrix.

**Input:** Two matrices, an nxm and mxp

**Output:** The resulting matrix nxp

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
    {1,  2,  3,  4},
    {5,  6,  7,  8},
    {9, 10, 11, 12}
  };

  std::cout << "m1*m2\n";
  std::cout << m1*m2 << std::endl;
}
```

**Output** from the lines above
```
m1
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

m2
|         -1         2         3         4 |
|          5        -6         7         8 |
|          9        10       -11        12 |

m1*m2
|         36        20       -16        56 |
|         75        38       -19       128 |
|        114        56       -22       200 |
```

_explanation of output_:

First m1 and m2 are displayed, then the product of m1 * m2 is displayed

**Implementation/Code:** The following is the code for Matrix Matrix Multiplication

This code uses [std::for_each](https://en.cppreference.com/w/cpp/algorithm/for_each) to initialize the result matrix to the correct size by sizing each row to the correct final size. Then the value of each element are calculated.

``` cpp
template <typename T, typename U, typename R = decltype(T() + U())>
Matrix<R> operator*(Matrix<T> const& m1, Matrix<U> const& m2)
{
  // check the sizes are compatible
  if (m1[0].size() != m2.size())
  {
    std::cerr << "ERROR: incorrectly sized matrices in mat * mat\n";
    exit(EXIT_FAILURE);
  }

  // initalize the result matrix
  Matrix<R> result(m1.size());
  std::for_each(begin(result), end(result), [&m2](std::vector<R>& row) {
    row.resize(m2[0].size());
  });

  // for each row of m1
  for (auto i = 0u; i < result.size(); ++i)
  {
    // for each column of m2
    for (auto j = 0u; j < result[0].size(); ++j)
    {
      // calculate the ij'th element of the result
      result[i][j] = 0;
      for (auto k = 0u; k < m2.size(); ++k)
      {
        result[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }

  // return the result
  return result;
}
```

**Last Modified:** October 2018
