---
title: Forward Substitution
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Forward Substitution Software Manual

**Routine Name:** forward_substitution

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./forwardSub.out** that can be executed.

**Description/Purpose:** `forwardSub` solves a linear system \\(Ly=b\\) for \\(y\\) by forward substitution where \\(L\\) is a lower triangular matrix.

**Input:**  takes two parameters, a lower triangular matrix and a vector:

```
@tparam T The type of the elements in L and b
@param L  A lower triangular matrix
@param b  A vector
```

**Output:** The routine returns the solution, y, to \\(Ly=b\\)

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double> L = {
    { 1,  0, 0, 0},
    { 5,  1, 0, 0},
    { 4, -6, 1, 0},
    {-4,  5, -9, 1}
  };

  std::vector<double> y{3, 5, -6, 8};

  auto b = L * y;

  std::cout << " L\n" << L << std::endl;
  std::cout << " b\n" << b << std::endl;
  std::cout << " Actual y\n" << y << std::endl;
  std::cout << " Calculated y\n";
  std::cout << forward_sub(L, b) << std::endl;
}
```

**Output** from the lines above

```
 L
|          1         0         0         0 |
|          5         1         0         0 |
|          4        -6         1         0 |
|         -4         5        -9         1 |

 b
[          3        20       -24        75 ]

 Real y
[          3         5        -6         8 ]

 Calculated y
[          3         5        -6         8 ]

```

_explanation of output_:

The matrix L is shown, then the constructed vetor b = L * y. Next the y vector that was used to construct b, and finally the calculated y vector. We can see that the actaul and caluclated y vectors match.

**Implementation/Code:** The following is the code for forward_substitution

``` cpp
template <typename T>
std::vector<T> forward_substitution(Matrix<T> L, std::vector<T> b)
{
  std::vector<T> y(b.size());

  for (auto i = 0u; i < L.size(); ++i)
  {
    T sum = 0.0;
    for (auto j = 0u; j < i; ++j)
    {
      sum += L[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }

  return y;
}
```

**Last Modification date:** October 2018
