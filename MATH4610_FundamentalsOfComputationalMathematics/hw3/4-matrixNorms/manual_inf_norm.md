---
title: Matrix Infinity Norm
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix Infinity Norm Software Manual

**Routine Name:** inf_norm

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixNorms.out** that can be executed.

**Description/Purpose:** This is a template function that can be used to calculate the \\(|| M ||_1\\) of any matrix s.t. \\(|| M ||_1 = \max_{1 \leq i \leq m} \sum_{i=1}^n |a_{ij}|\\), ie the max of the absolute row sums.

**Input:** The routine takes one argument, the matrix

```
@tparam The type of the elements in the matrix
@param  The matrix to take the norm of
```

**Output:** The function returns the infinity norm of the matrix

**Usage/Example:**

``` cpp
int main()
{
  Matrix<int> m = {
    {-3,  5,  7},
    { 2,  6,  4},
    { 0,  2,  8}
  };
  std::cout << m << std::endl;
  std::cout << "inf norm: " << inf_norm(m) << std::endl;
}
```

**Output** from the lines above
```
|         -3         5         7 |
|          2         6         4 |
|          0         2         8 |

inf norm: 15
```

_explanation of output_:

The first lines are the matrix

The second line is the inf norm of that matrix

**Implementation/Code:** The following is the code for inf_norm

The code finds the sum of the absolute values of the elements of the rows. First it uses [std::for_each](https://en.cppreference.com/w/cpp/algorithm/for_each) to iterate over the rows. Then for each row it uses [std::accumulate](https://en.cppreference.com/w/cpp/algorithm/accumulate) to add the absolute value of the elements of the rows  using [std::abs](https://en.cppreference.com/w/cpp/numeric/math/abs). It then uses [std::max_element](https://en.cppreference.com/w/cpp/algorithm/max_element) to identify the largest of the rowSums and returns it.

``` cpp
template <typename T>
T inf_norm(Matrix<T> m)
{
  // initialize the array to put the row sums
  std::vector<T> rowSums;
  rowSums.reserve(m[0].size());

  // for each row
  std::for_each(begin(m), end(m), [&colSums](auto const& row) {
    // sum the absolute values of the elements
    rowSums.push_back(std::accumulate(
      begin(row), end(row), 0.0, [](T acc, T e) { return acc + std::abs(e); }));
  });

  return *std::max_element(begin(rowSums), end(rowSums));
}
```

**Last Modified:** October 2018
