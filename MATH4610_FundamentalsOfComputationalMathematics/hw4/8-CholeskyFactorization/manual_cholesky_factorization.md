---
title: Cholesky Factorization
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Cholesky Factorization Software Manual

**Routine Name:** cholesky_factorization

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./cholesky.out** that can be executed.

**Description/Purpose:** Cholesky decomposition or Cholesky factorization is a decomposition of a Hermitian, positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose, which is useful for efficient numerical solutions. It was discovered by André-Louis Cholesky for real matrices. When it is applicable, the Cholesky decomposition is roughly twice as efficient as the LU decomposition for solving systems of linear equations.

**Input:** A square matrix A

**Output:** The lower triangular matrix such that L * L^T = A

**Usage/Example:**

``` cpp
int main()
{
  const auto n = 5;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;
  auto L = cholesky_factorization(A);
  auto LLT = L * transpose(L);

  std::cout << " A\n" << A << std::endl;
  std::cout << " L\n" << L << std::endl;
  std::cout << " L*L^T\n" << LLT << std::endl;
}
```

**Output** from the lines above
```
 A
|        172      37.7      64.2      -165      -143 |
|       37.7       204        42     -50.1      -114 |
|       64.2        42       108     -78.2     -33.1 |
|       -165     -50.1     -78.2       234       173 |
|       -143      -114     -33.1       173       185 |

 L
|       13.1         0         0         0         0 |
|       2.88        14         0         0         0 |
|        4.9         2      8.96         0         0 |
|      -12.6    -0.997     -1.63      8.51         0 |
|      -10.9     -5.91      3.61      4.11     0.833 |

 L*L^T
|        172      37.7      64.2      -165      -143 |
|       37.7       204        42     -50.1      -114 |
|       64.2        42       108     -78.2     -33.1 |
|       -165     -50.1     -78.2       234       173 |
|       -143      -114     -33.1       173       185 |
```

_explanation of output_:

First a random matrix A is generated, then Cholesky Factorization is applied and L is displayed. Finally L * L^T is displayed. We can see that A == L * L^T.

**Implementation/Code:** The following is the code for cholesky_factorization

``` cpp
template <typename T>
Matrix<T> cholesky_factorization(Matrix<T> const& A)
{
  // check for square matrix
  if (A.size() != A[0].size())
  {
    std::cerr << "ERROR: non-square matrix in cholesky_factorization"
              << std::endl;
  }

  // setup the result matrix
  Matrix<T> L(A.size());
  std::for_each(
    std::begin(L), std::end(L), [&](auto& row) { row.resize(A.size()); });

  int n = A.size();
  // for each row
  for (int i = 0; i < n; i++)
  {
    // for each column
    for (int j = 0; j < (i + 1); j++)
    {
      double s = 0;
      for (int k = 0; k < j; k++)
      {
        s += L[i][k] * L[j][k];
      }

      // diagonal
      if (i == j)
      {
        L[i][j] = std::sqrt(A[i][i] - s);
      }
      // non diagonal
      else
      {
        L[i][j] = 1.0 / L[j][j] * (A[i][j] - s);
      }
    }
  }

  // return the result
  return L;
}
```

**Last Modified:** October 2018
