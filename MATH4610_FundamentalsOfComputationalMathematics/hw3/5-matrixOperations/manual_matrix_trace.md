---
title:Matrix Transpose
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Transpose Software Manual

**Routine Name:** transpose

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixOpts.out** that can be executed.

**Description/Purpose:** The trace of an n-by-n square matrix A is defined to be the sum of the elements on the main diagonal (the diagonal from the upper left to the lower right). This code computes the trace of an nxn matrix.

**Input:** A nxn square matrix
```
@tparam T Type of the elements in the matrix
@param m  The matrix
```

**Output:** The trace of the matrix

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m1 = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
  };
  std::cout << m1 << std::endl;
  std::cout << trace(m1) << std::endl;
}
```

**Output** from the lines above
```
m1
|          1         2         3 |
|          4         5         6 |
|          7         8         9 |

trace m1
15
```

_explanation of output_:

First the matrix m1 is output. We can observe that the main diagonal is \\(1+5+9=15\\). Then the trace is output.

**Implementation/Code:** The following is the code for trace

First it checks to guarantee that the matrix is square. Then initialize the variable that will be used to gather the sum of the main diagonal. Finally add up the elements on the main diagonal and return the sum.

``` cpp
template <typename T>
T trace(Matrix<T> const& m)
{
  // check the matrix is square
  if (m.size() != m[0].size())
  {
    std::cerr << "ERROR: non square matrix in trace\n";
    exit(EXIT_FAILURE);
  }

  // initialize the trace to zero
  T t = 0;

  // sum the diagonal elements
  for (auto i = 0u; i < m.size(); ++i)
  {
    t += m[i][i];
  }

  // return the trace
  return t;
}
```

**Last Modified:** October 2018
