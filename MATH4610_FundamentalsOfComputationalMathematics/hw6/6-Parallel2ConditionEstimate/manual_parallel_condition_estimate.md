---
title: Parallel 2 Condition Number Matrix
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel 2 Condition Number Matrix Software Manual

**Routine Name:** parallel_condition_2_estimate

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./condition.out** that can be executed.

**Description/Purpose:** computes the 2 condition number of a matrix.

This code uses OpenMP to parallelize the power iteration and inverse power iteration functions in order to increase computation speed.

**Input:** A matrix A and a max number of iterations.

**Output:** The 2 condition of the matrix

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto condtitionNum = parallel_condition_2_estimate(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "2 Condition Number\n" << condtitionNum << std::endl;
}
```

**Output** from the lines above
```
A
|      -9.17      2.89     0.378      4.41      3.44 |
|       2.89      8.95     -3.28    -0.387      6.29 |
|      0.378     -3.28      8.94      -5.1      7.99 |
|       4.41    -0.387      -5.1      6.38      6.22 |
|       3.44      6.29      7.99      6.22      11.5 |

2 Condition Number
3.65
```

_explanation of output_:

First the matrix is output, then the condition number.

**Implementation/Code:** The following is the code for condition_2_estimate

As you can see, the code uses the parallel power and inverse power iteration functions previously written in 6.4 and 6.5

``` cpp
template <typename T>
T parallel_condition_2_estimate(Matrix<T> const& A, unsigned int const& MAX)
{
  auto maxEig = parallel_power_iteration(A, MAX);
  auto minEig = parallel_inverse_power_iteration(A, MAX);
  return std::abs(maxEig / minEig);
}
```

**Last Modified:** December 2018
