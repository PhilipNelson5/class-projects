---
title: Matrix Vector Multiplication
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Vector Multiplication Software Manual

**Routine Name:** Matrix Vector Multiplication

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** This routine overloads the `*` operator to multiply a matrix by a vector.

**Input:** This code takes two inputs, a matrix and a vector.

```
@tparam T Type of the elements in the matrix
@tparam U Type of the elements in the vector
@tparam R Type of the elements in the result vector
@param m  An mxn matrix
@param v  A vector with n elements
```

**Output:** A vector which contains the result of the matrix times the vector.

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m1 = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  std::vector<double> v1 = {1.1, 2.2, 3.3};
  std::cout << "m1*v1\n";
  std::cout << m1 * v1 << "\n";
}
```

**Output** from the lines above
```
m1
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

v1
[        1.1       2.2       3.3 ]

m1*v1
[       15.4      35.2        55 ]
```

_explanation of output_:

The first two block of output are the matrix m1 and the vector v1. Then the result of m1*v1.

**Implementation/Code:** The following is the code for Matrix Vector Multiplication

``` cpp
template <typename T, typename U, typename R = decltype(T() + U())>
std::vector<R> operator*(Matrix<T> const& m, std::vector<U> const& v)
{
  // check that the sizes are compatible
  if (m[0].size() != v.size())
  {
    std::cerr << "ERROR: incorrectly sized matrix or vector in mat * vec\n";
    exit(EXIT_FAILURE);
  }

  // initalize the result vector
  std::vector<R> result(m.size());

  // for every row of m
  for (auto i = 0u; i < m.size(); ++i)
  {
    R sum = 0;
    // for every element of v
    for (auto j = 0u; j < v.size(); ++j)
    {
      sum += m[i][j] * v[j];
    }
    result[i] = sum;
  }

  // return the result vector
  return result;
}
```

**Last Modified:** October 2018
