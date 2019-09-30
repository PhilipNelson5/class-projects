---
title: Matrix Infinity Norm
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Matrix Infinity Norm

**Routine Name:** infNorm

**Author:** Philip Nelson

**Language:** C++

## Description

`infNorm` calculates the infinity norm of a matrix. The infinity norm, \\(\|A\|_{\infty }\\) is the largest absolute sum of the rows of \\(A\\).

## Input

`infNorm(Matrix<T, M, N>& m)` requires:

* `Matrix<T, M, N> m` - an `M`x`N` matrix of type `T`

## Output

A double with the value of the infinity norm

## Code
{% highlight c++ %}
template <typename T, std::size_t M, std::size_t N>
T infNorm(Matrix<T, M, N>& m)
{
  std::array<T, N> rowSum;
  for (auto i = 0u; i < N; ++i)
    rowSum[i] = std::accumulate(
      m.begin(i), m.end(i), 0, [](T sum, T elem) { return sum + std::abs(elem); });

  return *std::max_element(rowSum.begin(), rowSum.end());
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<int, 3, 3> A({
                      {6, 1, 1},
                      {4, -2, 5},
                      {2, 8, 7}
                      });

  std::cout << " A\n" << A << std::endl;
  std::cout << "Infinity Norm: " << infNorm(A) << std::endl;
}
{% endhighlight %}

## Result
```
 A
|          6         1         1 |
|          4        -2         5 |
|          2         8         7 |

Infinity Norm: 17
```

**Last Modification date:** 10 February 2018
