---
title: Generate Square Symmetric Diagonally Dominant Matrix
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Generate Square Symmetric Diagonally Dominant Matrix Software Manual

**Routine Name:** Generate Square Symmetric Diagonally Dominant Matrix


**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./gen.out** that can be executed.

**Description/Purpose:** This code will initialize a square matrix so that the matrix has uniformly distributed numbers and is symmetric and diagonally dominant. The code  initializes the entries in the upper triangular and diagonal entries of the matrix to random numbers, reflects the values in the upper triangular part of the matrix into the lower triangular portion of the matrix, and adds a large enough entry (in magnitude) to the diagonal entry to ensure the matrix is diagonally dominant.

**Input:** The dimensionality of the matrix to be generated

**Output:** The square, uniformly distributed, symmetric, diagonally dominant matrix.

**Usage/Example:**

``` cpp
int main()
{
  auto m = generate_square_symmetric_diagonally_dominant_matrix(5);
  std::cout << " M\n" << m << std::endl;
}
```

**Output** from the lines above
```
|      -10.6     -4.21      2.48    -0.397     -1.41 |
|      -4.21     -9.11     -1.03     -6.73     -6.91 |
|       2.48     -1.03      6.93     0.521      -1.1 |
|     -0.397     -6.73     0.521       -13      5.07 |
|      -1.41     -6.91      -1.1      5.07     -16.3 |
```

_explanation of output_:

The generated matrix

**Implementation/Code:** The following is the code for 

``` cpp
Matrix<double> generate_square_symmetric_diagonally_dominant_matrix(
  unsigned int const n)
{
  // initialize the matrix
  Matrix<double> m(n);
  std::for_each(std::begin(m), std::end(m), [&](auto& row) { row.resize(n); });

  // initialize the upper triangle
  for (auto i = 0u; i < n; ++i)
  {
    for (auto j = i; j < n; ++j)
    {
      m[i][j] = random_double(-1e1, 1e1);
    }
  }

  // copy to the lower triangle
  for (auto i = 0u; i < n; ++i)
  {
    for (auto j = 0u; j < i; ++j)
    {
      m[i][j] = m[j][i];
    }
  }

  // enforce diagonal dominance
  for (auto i = 0u; i < n; ++i)
  {
    auto amax = *std::max_element(
      std::begin(m[i]), std::end(m[i]), [](auto const& e1, auto const& e2) {
        return std::abs(e1) < std::abs(e2);
      });

    if (amax != m[i][i])
    {
      m[i][i] = (amax > 0 ? amax + random_double(0, 1e1)
                          : amax - random_double(0, 1e1));
    }
  }

  return m;
}
```

**Last Modified:** October 2018
