---
title: Parallel Power Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel Power Iteration Software Manual

**Routine Name:** parallel_power-iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./parallelPowerIteration.out** that can be executed.

**Description/Purpose:** This code calculates the largest Eigenvalue of a `N`x`N` matrix by using the power method.

It uses openMP to parallelize the matrix vector multiplication in order to make the computation faster.

**Input:** The code takes an NxN matrix, A, and the number of iterations to run the method for.

**Output:** The larges eigenvalue of A. 

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto eigval = parallel_power_iteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Largest Eigenvalue\n" << eigval << std::endl;
}
```

**Output** from the lines above
```
A
|      -8.58     -6.33       1.1     -1.65     -4.79 |
|      -6.33      7.35     -3.03      7.28      -4.9 |
|        1.1     -3.03      12.8      6.87      4.23 |
|      -1.65      7.28      6.87        16      5.83 |
|      -4.79      -4.9      4.23      5.83      8.76 |

Largest Eigenvalue
25.2
```

_explanation of output_:

First the matrix A is displayed, then the larges eigenvalue found by the power method.

**Implementation/Code:** The following is the code for parallel_power_method

The code uses the parallel matrix vector multiply previously written in assignment 3.7

``` cpp
template <typename T>
T parallel_power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v_k(A.size());

  random_double_fill(std::begin(v_k), std::end(v_k), -100, 100);

  for (auto i = 0u; i < MAX; ++i)
  {
    v_k = parallel_multiply(A, v_k);
    v_k = v_k / p_norm(v_k, 2);
  }

  auto pointwise = v_k * parallel_multiply(A, v_k);
  auto lambda = std::accumulate(
    std::begin(pointwise), std::end(pointwise), T(0.0), [](auto acc, auto val) {
      return acc + val;
    });

  return lambda;
}
```

**Last Modified:** December 2018
