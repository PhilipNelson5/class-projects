---
title: 2 Condition Number Matrix
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# 2 Condition Number Matrix Software Manual

**Routine Name:** condition_2_estimate

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./condition.out** that can be executed.

**Description/Purpose:** computes the 2 condition number of a matrix.

**Input:** A matrix A and a max number of iterations.

**Output:** The 2 condition of the matrix

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto conditionNum = condition_2_estimate(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "2 Condition Number\n" << conditionNum << std::endl;
}
```

**Output** from the lines above
```
|       10.5     -8.18      9.53     -5.89      -3.8 |
|      -8.18     -12.7      6.07      3.72     -4.48 |
|       9.53      6.07      9.62     -5.72      4.21 |
|      -5.89      3.72     -5.72     -11.9      4.24 |
|       -3.8     -4.48      4.21      4.24     -8.68 |

2 Condition Number
4.46
```

_explanation of output_:

First the matrix is output, then the condition number.

**Implementation/Code:** The following is the code for condition_2_estimate

``` cpp
template <typename T>
T condition_2_estimate(Matrix<T> const& A, unsigned int const & MAX)
{
  auto maxEig = power_iteration(A, MAX);
  auto minEig = inverse_power_iteration(A, MAX);
  return std::abs(maxEig / minEig);
}
```

**Last Modified:** December 2018
