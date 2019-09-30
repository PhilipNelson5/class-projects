---
title: LU Factorization
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# LU Factorization

**Routine Name:** luFactorize

**Author:** Philip Nelson

**Language:** C++

## Description

`luFactorize` turns a matrix into upper and lower triangular matrices such that \\(LU=PA\\)

## Input

`luFactorize()` takes not arguments but must be called by a `Matrix<T, M, M> of type `T` and size `MxM`

## Output

`luFactorize` returns a `std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>>` with \\(L,\ U,\ P\\) matricies. Destructured returning is recommended.

## Code
{% highlight c++ %}
std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>> luFactorize()
{
  auto I = identity<T, N>();
  auto P = I;
  Matrix<T, N, N> L(0);
  Matrix<T, N, N> U(m);
  for (auto j = 0u; j < N; ++j) // columns
  {
    auto largest = U.findLargestInCol(j, j);
    if (largest != j)
    {
      L.swapRows(j, largest);
      U.swapRows(j, largest);
      P.swapRows(j, largest);
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
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 5, 5> A(1, 10); // random 5x5 with values from 0-10
  auto [L, U, P] = A.luFactorize();

  std::cout << " L\n" << L << std::endl;
  std::cout << " U\n" << U << std::endl;
  std::cout << " P\n" << P << std::endl << std::endl;
  std::cout << " LU\n" << L*U << std::endl;
  std::cout << " PA\n" << P*A << std::endl;
}
{% endhighlight %}

## Result
```
  A
|         8        8        4        2        6 |
|         5        5        5        3        1 |
|        10        3       10        3        3 |
|         5        2        9        4        8 |
|        10        3        7        7        4 |

 L
|         1        0        0        0        0 |
|       0.8        1        0        0        0 |
|       0.5  0.08929        1        0        0 |
|         1        0  -0.6885        1        0 |
|       0.5    0.625   0.5738  0.05136        1 |

 U
|        10        3       10        3        3 |
|         0      5.6       -4     -0.4      3.6 |
|         0        0    4.357    2.536    6.179 |
|         0        0        0    5.746    5.254 |
|         0        0        0        0   -6.565 |

 P
|         0        0        1        0        0 |
|         1        0        0        0        0 |
|         0        0        0        1        0 |
|         0        0        0        0        1 |
|         0        1        0        0        0 |


 LU
|        10        3       10        3        3 |
|         8        8        4        2        6 |
|         5        2        9        4        8 |
|        10        3        7        7        4 |
|         5        5        5        3        1 |

 PA
|        10        3       10        3        3 |
|         8        8        4        2        6 |
|         5        2        9        4        8 |
|        10        3        7        7        4 |
|         5        5        5        3        1 |

```

**Last Modification date:** 07 February 2018
