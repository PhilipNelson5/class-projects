---
title: Power Iteration
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Power Iteration

**Routine Name:** powerIteration

**Author:** Philip Nelson

**Language:** C++

## Description

`powerIteration` calculates the largest Eigenvalue of a `N`x`N` matrix.

## Input


`powerIteration(Matrix<T, N, N> const& A, unsigned int const& MAX)` requires:

* `Matrix<T, N, N> const& A` - an `N`x`N` matrix
* `unsigned int const& MAX` - the maximum number of iterations

## Output

The largest Eigenvalue

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
T powerIteration(Matrix<T, N, N> const& A, unsigned int const& MAX)
{
  std::array<T, N> b_k;

  for (auto&& e : b_k)
    e = randDouble(0.0, 10.0);

  for (auto i = 0u; i < MAX; ++i)
  {
    auto Ab_k = A * b_k;
    auto norm = pNorm(Ab_k, 2);
    b_k = Ab_k / norm;
  }
  return pNorm(A * b_k, 2);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 5, 5> A(
    [](unsigned int const& i, unsigned int const& j) { return 1.0 / (i + j + 1.0); });

  auto eigval = powerIteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Largest Eigenvalue\n" << eigval << std::endl;
}
{% endhighlight %}

## Result
```
A
|          1       0.5     0.333      0.25       0.2 |
|        0.5     0.333      0.25       0.2     0.167 |
|      0.333      0.25       0.2     0.167     0.143 |
|       0.25       0.2     0.167     0.143     0.125 |
|        0.2     0.167     0.143     0.125     0.111 |

Largest Eigenvalue
1.57

```

**Last Modification date:** 27 February 2018
