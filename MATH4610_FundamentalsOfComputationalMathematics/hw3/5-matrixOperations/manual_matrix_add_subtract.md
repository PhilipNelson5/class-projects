---
title: Matrix Addition and Subtraction
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Addition Software Manual

**Routine Name:** Matrix Addition and Subtraction

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** This routine overloads the `+` and `-` operators in c++ allowing two matrices to be added or subtracted with the following syntax, `a + b` and `a - b`.

**Input:** The operator requires two operands, `a` and `b`, where `a, b` are `std::vector<std::vector<T>>`

```
@tparam T Type of the elements in the first matrix
@tparam U Type of the elements in the second matrix
@tparam R Type of the elements in the result matrix
@param a  The first matrix
@param b  The second matrix
```

**Output:** A matrix with the result of matrix addition or subtraction with the two matrix operands.

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m1 = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  std::cout << m1 + m1 << std::endl;
  std::cout << m1 - m1 << std::endl;
}
```

**Output** from the lines above
```
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

|          2         4         6 |
|          8        10        12 |
|         14        16        18 |

|          0         0         0 |
|          0         0         0 |
|          0         0         0 |
```

_explanation of output_:

The matrix is `m1`.

The second matrix is the result of `m1 + m1`.

The third matrix is the result of `m1 - m1`.

**Implementation/Code:** The following is the code for Matrix addition and subtraction

For code re-usability, the implementation takes advantage of the c++ preprocessor to generate the actual addition and subtraction code. The difference between adding and subtracting matrices is simply the difference between using the plus operator or the minus operator. Hence, `matrix_add_subtract` defines the generic form of the operation with the operator replaced with a variable. This way `matrix_add_subtract(+) matrix_add_subtract(-)` can fill in the operator and the preprocessor will generate the code, no branching structure or duplicate code required.

``` cpp
#define matrix_add_subtract(op)                                                \
  template <typename T, typename U, typename R = decltype(T() + U())>          \
  Matrix<R> operator op(Matrix<T> const& a, Matrix<U> const& b)                \
  {                                                                            \
    // check sizes are compatible                                              \
    if (a.size() != b.size() || a[1].size() != b[0].size())                    \
    {                                                                          \
      std::cerr << "ERROR: bad size in matrix_add_subtract\n";                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
                                                                               \
    // initalize the result matrix                                             \
    Matrix<R> result(a.size());                                                \
                                                                               \
    // for every row in a                                                      \
    for (auto i = 0u; i < a.size(); ++i)                                       \
    {                                                                          \
      // initalize the next row of the result                                  \
      result[i].reserve(a[i].size());                                          \
                                                                               \
      // for every column of a                                                 \
      for (auto j = 0u; j < a[i].size(); ++j)                                  \
      {                                                                        \
        // push the ij'th elements of a and b added or subtracted              \
        result[i].push_back(a[i][j] op b[i][j]);                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    // return the result matrix                                                \
    return result;                                                             \
  }

// call the macro with the addition and subtraction operator
matrix_add_subtract(+) matrix_add_subtract(-)
```

**Last Modified:** October 2018
