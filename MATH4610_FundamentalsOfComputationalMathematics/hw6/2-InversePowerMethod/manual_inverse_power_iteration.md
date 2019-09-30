---
title: Inverse Power Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Inverse Power Iteration Software Manual

**Routine Name:** inverse_power_iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./inverseIteration.out** that can be executed.

**Description/Purpose:** The code calculates the smalled Eigenvalue of an NxN matrix by the inverse power method.

**Input:** The code takes an NxN matrix A, and the max number of iterations

**Output:** The smallest Eigenvalue

**Usage/Example:**

``` cpp
int main()
{
  auto A = generate_square_symmetric_diagonally_dominant_matrix(5u);

  auto eigval = inverse_power_iteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Smallest Eigenvalue\n" << eigval << std::endl;
}
```

**Output** from the lines above
```
A
|       17.9     -3.94      8.13     -6.44       8.3 |
|      -3.94     -14.3     -6.75      7.63     -8.81 |
|       8.13     -6.75      14.2      6.61     -7.91 |
|      -6.44      7.63      6.61      8.36     -3.67 |
|        8.3     -8.81     -7.91     -3.67       -15 |

Smallest Eigenvalue
1.98
```

_explanation of output_:

First the matrix A is displayed, then the smalled eigenvalue is displayed. 

**Implementation/Code:** The following is the code for inverse_power_iteration

``` cpp
template <typename T>
T inverse_power_iteration(Matrix<T> const& A, unsigned int const& MAX)
{
  std::vector<T> v(A.size());

  random_double_fill(std::begin(v), std::end(v), -100, 100);

  T lamda = 0.0;
  for (auto i = 0u; i < MAX; ++i)
  {
    auto w = solve_linear_system_LU(A, v);
    v = w / p_norm(w, 2);
    auto pointwise = v * (A * v);
    lamda = std::accumulate(std::begin(pointwise),
                            std::end(pointwise),
                            T(0.0),
                            [](auto acc, auto val) { return acc + val; });
  }
  return lamda;
}
```

**Last Modified:** December 2018
