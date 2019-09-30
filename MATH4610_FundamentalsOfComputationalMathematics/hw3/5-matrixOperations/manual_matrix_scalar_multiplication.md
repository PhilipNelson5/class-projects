---
title: Matrix Scalar Multiplication
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Scalar Multiplication Software Manual

**Routine Name:** Matrix Scalar Multiplication

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** This routine overloads the `*` operator to multiply a matrix by a scalar and a scalar by a matrix. When multiplying scalars and matrices, each element of the matrix is scaled by the scalar.

**Input:** The code takes two parameters, a matrix m and a scalar s. The parameters can be supplied in either order.

```
@tparam T Type of the elements in the matrix
@tparam S Type of the elements in the vector
@tparam R Type of the elements in the result vector
@param s  A scalar value
@param m  An mxn matrix
```

**Output:** The matrix scaled by the scalar

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m1 = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  std::cout << "m1*3\n";
  std::cout << m1 * 3 << std::endl;
  std::cout << "3*m1\n";
  std::cout << 3 * m1 << std::endl;
}
```

**Output** from the lines above
```
m1
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

m1*3
|          3         6         9 |
|         12        15        18 |
|         21        24        27 |

3*m1
|          3         6         9 |
|         12        15        18 |
|         21        24        27 |
```

_explanation of output_:

The first two block of output are the matrix m1 and the vector v1. Then the result of m1*v1.

**Implementation/Code:** The following is the code for Matrix Scalar Multiplication

This code uses [std::for_each](https://en.cppreference.com/w/cpp/algorithm/for_each) to iterate through each row and column of the matrix and multiply it by the given scalar `s`.The result is then returned.

``` cpp
template <typename T, typename S, typename R = decltype(S() * T())>
Matrix<R> operator*(S const s, Matrix<T> const& m)
{
  // initialize the result matrix to the passed in matrix
  Matrix<R> result = m;

  // for each row
  std::for_each(std::begin(result), std::end(result), [&](auto& row) {
    // for each column
    std::for_each(std::begin(row), std::end(row), [&](auto& elem) {
      // multiply the element by s
      elem *= s;
    });
  });

  // return the result
  return result;
}

// this just lets you multiply s * m as well as m * s
template <typename T, typename S, typename R = decltype(T() * S())>
inline Matrix<R> operator*(Matrix<T> const& m, S const s)
{
  return s * m;
}
```

**Last Modified:** October 2018
