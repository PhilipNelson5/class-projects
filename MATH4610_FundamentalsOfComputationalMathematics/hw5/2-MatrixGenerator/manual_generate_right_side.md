---
title: Generate Right Side
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Generate Right Side Software Manual

**Routine Name:** Generate Right Side

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./gen.out** that can be executed.

**Description/Purpose:** This code will generate the right hand side of a system of equations by multiplying the matrix above into a vector of ones

**Input:** A matrix

**Output:** A vector which is the result of multiplying the matrix above into a vector of ones

**Usage/Example:**

``` cpp
int main()
{
  auto m = generate_square_symmetric_diagonally_dominant_matrix(5);
  auto b = generate_right_side(m);
  std::cout << " M\n" << m << std::endl;
  std::cout << " b\n" << b << std::endl;
}
```

**Output** from the lines above
```
 M
|      -10.6     -4.21      2.48    -0.397     -1.41 |
|      -4.21     -9.11     -1.03     -6.73     -6.91 |
|       2.48     -1.03      6.93     0.521      -1.1 |
|     -0.397     -6.73     0.521       -13      5.07 |
|      -1.41     -6.91      -1.1      5.07     -16.3 |

 b
[      -14.1       -28       7.8     -14.6     -20.6 ]
```

_explanation of output_:

A matrix and the produced right side

**Implementation/Code:** The following is the code for generate_right_side

``` cpp
template <typename T>
inline std::vector<T> generate_right_side(Matrix<T> m)
{
  return m * std::vector<T>(m.size(), 1);
}
```

**Last Modified:** October 2018
