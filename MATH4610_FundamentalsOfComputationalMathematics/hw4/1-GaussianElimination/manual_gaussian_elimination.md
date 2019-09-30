---
title: Gaussian Elimination
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Gaussian Elimination Software Manual

**Routine Name:** Gaussian Elimination

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./gaussianEliminationout** that can be executed.

**Description/Purpose:** Gaussian elimination (also known as row reduction) is an algorithm for solving systems of linear equations. This routine accepts a square matrix and modifies it to be in a row reduced form with the option to reduce to have one's on the diagonal.

**Input:** The routine takes a matrix to be row reduced and optionally a boolean, if true the matrix will be reduced to have ones on the diagonal.

```
@paramt T   Type of the elements in the matrix
@param mat  The matrix
@param ones Should the diagonal be ones
```

**Output:** The routine has no output, all the work is done in place.

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double> m1 = {
    { 1, -1,  2, -1},
    { 4,  4, -2,  1},
    {-3,  5, -7, 12},
    {-2,  2, -4,  4}
  };
  std::cout << "m1\n" << m1 << std::endl;
  gaussian_emlimination(m1);
  std::cout << m1 << std::endl;
}
```

**Output** from the lines above
```
m1
|          1        -1         2        -1 |
|          4         4        -2         1 |
|         -3         5        -7        12 |
|         -2         2        -4         4 |

|          4         4        -2         1 |
|          0         8      -8.5      12.8 |
|          0         0     0.375      1.94 |
|          0         0         0         2 |
```

_explanation of output_:

First, the matrix is displayed, then the matrix after row reduction.

**Implementation/Code:** The following is the code for gaussian_elimination

``` cpp
template <typename T>
void gaussian_emlimination(Matrix<T>& mat, bool ones = false)
{
  int m = mat.size();    // row dimensionality
  int n = mat[0].size(); // col dimensionality
  int h = 0;             // row pivot
  int k = 0;             // col pivot

  while (h < m && k < n)
  {
    // find the next pivot
    auto piv = h;
    auto max = abs(mat[h][k]);
    for (auto i = h + 1; i < m; ++i)
    {
      if (max < abs(mat[i][k]))
      {
        max = abs(mat[i][k]);
        piv = i;
      }
    }

    if (mat[piv][k] == 0) // no pivot in the col
    {
      ++k; // go to next col
      continue;
    }

    std::swap(mat[h], mat[piv]); // swap pivot row with current row

    // for all rows below pivot
    for (auto i = h + 1; i < m; ++i)
    {
      auto f = mat[i][k] / mat[h][k];
      mat[i][k] = 0; // zero out the rest of the col

      // for all the rest of the elements in the row
      for (auto j = k + 1; j < n; ++j)
      {
        mat[i][j] = mat[i][j] - mat[h][j] * f;
      }
    }

    ++h, ++k; // increase current row and col
  }

  if (ones) // make ones down the diagonal
  {
    for (auto i = 0; i < m; ++i)
    {
      if (mat[i][i] == 0) break;
      for (auto j = i + 1; j < n; ++j)
      {
        mat[i][j] /= mat[i][i];
      }
      mat[i][i] = 1;
    }
  }
}

```

**Last Modified:** October 2018
