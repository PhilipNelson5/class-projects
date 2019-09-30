---
title: Vector Outer Product
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Vector Outer Product Software Manual

**Routine Name:** outer_product

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorOpts2.out** that can be executed.

**Description/Purpose:**
The outer product is equivalent to a matrix multiplication \\(uv^T\\), provided that u is represented as a m × 1 column vector and v as a n × 1 column vector (which makes \\(v^T\\) a row vector).

**Input:** Two vectors

```
@tparam T The type of the elements in v1 and v2
@param v1 The first vector
@param v2 The second vector
```

**Output:** A matrix which is the result of the outer product.

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> v1{1, 2, 3, 4};
  std::vector<double> v2{1, 2, 3, 4};
  std::cout << "v1\n" << v1 << '\n' << "v2\n" << v2 << "\nv1 * v2^t";
  std::cout << outer_product(v1, v2) << '\n';
}
```

**Output** from the lines above
```
v1
[          1         2         3         4 ]

v2
[          1         2         3         4 ]

v1 * v2^t
|          1         2         3         4 |
|          2         4         6         8 |
|          3         6         9        12 |
|          4         8        12        16 |
```

_explanation of output_:

The first two lines are the two vectors to be used in the outer product

Then the result of the outer product is displaced

**Implementation/Code:** The following is the code for outer_product

``` cpp
template <typename T>
Matrix<T> outer_product(std::vector<T> v1, std::vector<T> v2)
{
  // check that vectors are the same size
  if (v1.size() != v2.size())
  {
    std::cerr << "ERROR: bad size in Determinant\n";
    exit(EXIT_FAILURE);
  }

  // setup resultant matrix
  Matrix<T> m(v1.size());
  std::for_each(
    std::begin(m), std::end(m), [&](auto& row) { row.resize(v2.size()); });

  // m_{ij} = v1_i * v2_j
  for (auto i = 0u; i < v2.size(); ++i)
  {
    for (auto j = 0u; j < v1.size(); ++j)
    {
      m[i][j] = v1[i] * v2[j];
    }
  }

  return m;
}
```

**Last Modified:** October 2018
