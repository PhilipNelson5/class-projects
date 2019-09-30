---
title: Matrix One Norm
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Matrix One Norm

**Routine Name:** oneNorm

**Author:** Philip Nelson

**Language:** C++

## Description

`oneNorm` calculates the one norm of a matrix. Give a matrix \\(A\\), the one norm, \\(\|A\|_{1}\\), is the maximum of the absolute column sums.

## Input


`oneNorm(Matrix<T, M, N>& m)` requires:

* `Matrix<T, M, N> m` - an `M`x`N` matrix of type `T`

## Output

A double with the desired one norm of the matrix.

## Code
{% highlight c++ %}
template <typename T, std::size_t M, std::size_t N>
T oneNorm(Matrix<T, M, N>& m)
{
  std::array<T, N> colSum;
  for (auto j = 0u; j < M; ++j)
  {
    colSum[j] = 0;
    for (auto i = 0u; i < N; ++i)
    {
      colSum[j] += std::abs(m[i][j]);
    }
  }

  return *std::max_element(colSum.begin(), colSum.end());
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
std::cout << "One Norm: " << oneNorm(A) << std::endl;
}
{% endhighlight %}

## Result
```
 A
|          6         1         1 |
|          4        -2         5 |
|          2         8         7 |

One Norm: 13
```

**Last Modification date:** 10 February 2018
