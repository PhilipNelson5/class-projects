---
title: Parallel Inverse Power Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Parallel Inverse Power Iteration Software Manual

**Routine Name:** parallel_inverse_power_iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./inverseIteration.out** that can be executed.

**Description/Purpose:** The code calculates the smalled Eigenvalue of an NxN matrix by the inverse power method.

The code uses OpenMP to parallelize the matrix vector multiplication in order to increase performance.

**Input:** The code takes an NxN matrix A, and the max number of iterations

**Output:** The smallest Eigenvalue

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto eigval = parallel_inverse_power_iteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Smallest Eigenvalue\n" << eigval << std::endl;
}
```

**Output** from the lines above
```
A
|       9.64      2.78     -8.89     -6.95      1.78 |
|       2.78      13.2      8.44     -5.49       4.5 |
|      -8.89      8.44     -16.7      -1.9     -5.27 |
|      -6.95     -5.49      -1.9     -13.7       4.6 |
|       1.78       4.5     -5.27       4.6       8.1 |

Smallest Eigenvalue
9.25
```

_explanation of output_:

First the matrix A is displayed, then the smalled eigenvalue is displayed. 

**Implementation/Code:** The following is the code for parallel_inverse_power_iteration

The code uses the parallel matrix vector multiply previously written in assignment 3.7

``` cpp
template <typename T>
T parallel_inverse_power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v(A.size());

  random_double_fill(std::begin(v), std::end(v), -100, 100);

  T lamda = 0.0;
  for (auto i = 0u; i < MAX; ++i)
  {
    auto w = solve_linear_system_LU(A, v);
    v = w / p_norm(w, 2);
    auto pointwise = v * parallel_multiply(A, v);
    lamda = std::accumulate(std::begin(pointwise),
                            std::end(pointwise),
                            T(0.0),
                            [](auto acc, auto val) { return acc + val; });
  }
  return lamda;
}
```

**Last Modified:** December 2018
