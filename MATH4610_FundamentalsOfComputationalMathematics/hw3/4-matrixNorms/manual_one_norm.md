---
title: Matrix One Norm
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Matrix One Norm Software Manual

**Routine Name:** one_norm

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./matrixNorms.out** that can be executed.

**Description/Purpose:** This is a template function that can be used to calculate the \\(\| M \|_1\\) of any matrix s.t. \\(\| M \|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^m |a_{ij}|\\), ie the max of the absolute column sums.

**Input:** The routine takes one argument, the matrix

```
@tparam The type of the elements in the matrix
@param  The matrix to take the norm of
```

**Output:** The function returns the one norm of the matrix

**Usage/Example:**

``` cpp
int main()
{
  std::vector<std::vector<int>> m = {
    {-3, 5, 7},
    {2, 6, 4},
    {0, 2, 8}
  };
  std::cout << m << std::endl;
  std::cout << "one norm: " << one_norm(m) << std::endl;
}
```

**Output** from the lines above
```
|         -3         5         7 |
|          2         6         4 |
|          0         2         8 |

one norm: 19
```

_explanation of output_:

The first lines are the matrix

The second line is the one norm of that matrix

**Implementation/Code:** The following is the code for one_norm

The code finds the sum of the absolute values of the columns using [std::abs](https://en.cppreference.com/w/cpp/numeric/math/abs). It then uses [std::max_element](https://en.cppreference.com/w/cpp/algorithm/max_element) to identify the 

``` cpp
template <typename T>
T one_norm(Matrix<T> m)
{
  // initalize the array to hold the sums of the columns
  std::vector<T> colSums;
  colSums.reserve(m[0].size());

  // for every column
  for (auto j = 0u; j < m[0].size(); ++j)
  {
    // for every element column wise
    T sum = 0;
    for (auto i = 0u; i < m.size(); ++i)
    {
      // sum the absolute values of the column elements
      sum += std::abs(m[i][j]);
    }
    // push the column sum on the end of the colSum array
    colSums.push_back(sum);
  }

  // find and return the maximum element of the column sums
  return *std::max_element(begin(colSums), end(colSums));
}
```

**Last Modified:** October 2018
