---
title: LU Factorization
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# LU Factorization Software Manual

**Routine Name:** LU_factorize

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./luf.out** that can be executed.

**Description/Purpose:** Lower–upper (LU) decomposition or factorization factors a matrix as the product of a lower triangular matrix and an upper triangular matrix. The product sometimes includes a permutation matrix as well such that \\(LU=PA\\). LU decomposition can be viewed as the matrix form of Gaussian elimination. Computers usually solve square systems of linear equations using LU decomposition, and it is also a key step when inverting a matrix or computing the determinant of a matrix. LU decomposition was introduced by mathematician Tadeusz Banachiewicz in 1938.

**Input:** A matrix.

```
@tparam T The type of the elements of A
@param m  The matrix to be decomposed
```

**Output:** The L U P component matrices.

```
@return A tuple composed of the decomposed matrix m into 
L - Lower triangular
U - Upper triangular
P - Permutation matrix
```

**Usage/Example:**

``` cpp
int main()
{
  Matrix<double> A = {
    {1, 2, 3, 4},
    {4, 5, 6, 6},
    {2, 5, 1, 2},
    {7, 8, 9, 7}
  };
  auto [L, U, P] = LU_factorize(A);
  std::cout << " A\n" << A << std::endl;
  std::cout << " L\n" << L << std::endl;
  std::cout << " U\n" << U << std::endl;
  std::cout << " P\n" << P << std::endl;
  std::cout << " LU\n" << L * U << std::endl;
  std::cout << " PA\n" << P * A << std::endl;
}
```

**Output** from the lines above
```
 A
|          1         2         3         4 |
|          4         5         6         6 |
|          2         5         1         2 |
|          7         8         9         7 |

 L
|          1         0         0         0 |
|      0.286         1         0         0 |
|      0.143     0.316         1         0 |
|      0.571     0.158       0.5         1 |

 U
|          7         8         9         7 |
|          0      2.71     -1.57         0 |
|          0  1.11e-16      2.21         3 |
|          0 -2.47e-32         0       0.5 |

 P
|          0         0         0         1 |
|          0         0         1         0 |
|          1         0         0         0 |
|          0         1         0         0 |

 LU
|          7         8         9         7 |
|          2         5         1         2 |
|          1         2         3         4 |
|          4         5         6         6 |

 PA
|          7         8         9         7 |
|          2         5         1         2 |
|          1         2         3         4 |
|          4         5         6         6 |
```

_explanation of output_:

First we see the matrix A, then the decomposed L, U, and P matrices. Next is the result of multiplying L and U. Finally is the result of multiplying P and A. We can see that LU == PA so the decomposition was successful.

**Implementation/Code:** The following is the code for LU_factorize

For ease of use, LU_factorize returns a [std::tuple](https://en.cppreference.com/w/cpp/utility/tuple) of matrices, three matrices grouped together. Using [structured binding](https://en.cppreference.com/w/cpp/language/structured_binding) can take advantage of this technique as seen in the example code. This makes for cleaner and more readable code.

``` cpp
template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> LU_factorize(Matrix<T> m)
{
  if (m.size() != m[0].size())
  {
    std::cerr << "ERROR: Non square matrix in luFactorize\n";
    exit(EXIT_FAILURE);
  }

  auto N = m.size();

  auto I = identity<T>(N);
  auto P = I;

  Matrix<T> L = zeros<T>(N);
  Matrix<T> U = m;

  for (auto j = 0u; j < N; ++j) // columns
  {
    auto largest = findLargestInCol(U, j, j);

    if (largest != j)
    {
      std::swap(L[j], L[largest]);
      std::swap(U[j], U[largest]);
      std::swap(P[j], P[largest]);
    }

    auto pivot = U[j][j];
    auto mod = I;

    for (auto i = j + 1; i < N; ++i) // rows
    {
      mod[i][j] = -U[i][j] / pivot;
    }

    L = L + I - mod;
    U = mod * U;
  }

  L = I + L;

  return {L, U, P};
}
```

**Last Modified:** October 2018
