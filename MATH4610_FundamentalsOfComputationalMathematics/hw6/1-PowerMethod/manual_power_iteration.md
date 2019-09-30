---
title: Power Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Power Iteration Software Manual

**Routine Name:** power-iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./powerIteration.out** that can be executed.

**Description/Purpose:** This code calculates the largest Eigenvalue of a `N`x`N` matrix by using the power method.

**Input:** The code takes an NxN matrix, A, and the number of iterations to run the method for.

**Output:** The larges eigenvalue of A. 

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto eigval = power_iteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Largest Eigenvalue\n" << eigval << std::endl;
}
```

**Output** from the lines above
```
A
|       10.2     -3.01      9.58      -5.4      7.28 |
|      -3.01      11.8       6.1      5.54      6.94 |
|       9.58       6.1      12.1      5.53      5.25 |
|       -5.4      5.54      5.53      8.09      3.99 |
|       7.28      6.94      5.25      3.99      9.36 |

Largest Eigenvalue
29.1
```

_explanation of output_:

First the matrix A is displayed, then the larges eigenvalue found by the power method.

**Implementation/Code:** The following is the code for power_method

``` cpp
template <typename T>
T power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v_k(A.size());

  random_double_fill(std::begin(v_k), std::end(v_k), -100, 100);

  for (auto i = 0u; i < MAX; ++i)
  {
    v_k = A * v_k;
    v_k = v_k / p_norm(v_k, 2);
  }

  auto pointwise = v_k * (A * v_k);
  auto lambda = std::accumulate(
    std::begin(pointwise), std::end(pointwise), T(0.0), [](auto acc, auto val) {
      return acc + val;
    });
  return lambda;
}
```

**Last Modified:** December 2018
