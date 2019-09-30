---
title: Cholesky Factorization test
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Cholesky Factorization Software Manual

**Name:** Cholesky factorization test

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./choleskyTest.out** that can be executed.

**Description/Purpose:** This is a demonstration of Cholesky factorization on a large matrix


This program generates a random matrix of 1000 x 1000 elements. It then multiplies the generated matrix by it's transpose to ensure it is a symmetric, positive definite matrix. This matrix, A, then has Cholesky factorization performed on it. The Cholesky factorization is timed. Then the original matrix is then computed from the factorized matrix L by multiplying L by L transpose. This should recreate the original matrix A. Finally the time is printed to the screen and the and the original matrix, A, is compared to L\*L^T.

``` cpp
int main()
{
  const auto n = 1000;

  Matrix<double> C = rand_double_NxM(n, n, -10, 10);

  auto A = transpose(C) * C;

  auto start = std::chrono::high_resolution_clock::now();
  auto L = cholesky_factorization(A);
  auto end = std::chrono::high_resolution_clock::now();

  auto LLT = L * transpose(L);

  auto result = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Cholesky factorization completed in " << result << " ms"
            << std::endl;

  if (allclose(A, LLT, 1e-10))
    std::cout << "A == LLT" << std::endl;
  else
    std::cout << "A =/= LLT" << std::endl;
}
```

**Output** from the lines above
```
Cholesky factorization completed in 246.264 ms
A == LLT
```

_explanation of output_:

We can see that A is equal to L\*L^T and it took 246 milliseconds to compute the Cholesky Factorization

**Last Modified:** December 2018
