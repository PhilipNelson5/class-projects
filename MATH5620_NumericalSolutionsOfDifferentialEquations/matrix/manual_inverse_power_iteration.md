---
title: Inverse Power Iteration
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Power Iteration

**Routine Name:** inversePowerIteration

**Author:** Philip Nelson

**Language:** C++

## Description

`inversePowerIteration` calculates the smallest Eigenvalue of a `N`x`N` matrix.

## Input

`inversePowerIteration(Matrix<T, N, N> const& A, unsigned int const& MAX)` requires:

* `Matrix<T, N, N> const& A` - an `N`x`N` matrix
* `unsigned int const& MAX` - the maximum number of iterations

## Output

The smallest Eigenvalue

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
T inversePowerIteration(Matrix<T, N, N> & A, unsigned int const& MAX)
{
  std::array<T, N> v;
  for (auto&& e : v)
    e = randDouble(0.0, 10.0);

  T lamda = 0;
  for (auto i = 0u; i < MAX; ++i)
  {
    auto w = A.solveLinearSystemLU(v);
    v = w / pNorm(w,2);
    lamda = v*(A*v);
  }
    return lamda;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  Matrix<double, 5, 5> A(
    [](unsigned int const& i, unsigned int const& j) { return 1.0 / (i + j + 1.0); });

  auto eigval = inversePowerIteration(A, 1000u);
  std::cout << "A\n" << A << std::endl;
  std::cout << "Smallest Eigenvalue\n" << eigval << std::endl;
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

Smallest Eigenvalue
3.29e-06

```

**Last Modification date:** 27 February 2018
