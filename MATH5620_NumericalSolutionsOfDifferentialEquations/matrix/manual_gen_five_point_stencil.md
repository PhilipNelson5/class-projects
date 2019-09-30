---
title: Five Point Stencil
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Five Point Stencil

**Routine Name:** fivePointStencil

**Author:** Philip Nelson

**Language:** C++

## Description

`fivePointStencil` returns an \\(N^2\\)x\\(N^2\\) matrix for the five point stencil for the 2D finite difference method

## Input

`Matrix<T, N * N, N * N> fivePointStencil()` takes no input but requires the size and template parameters \\(N\\) and \\(T\\)

## Output

A \\(N^2\\)x\\(N^2\\) matrix with the stencil

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
Matrix<T, N * N, N * N> fivePointStencil()
{
  return Matrix<T, N * N, N * N>([](int i, int j) {
    if (i == j) return -4;
    if (i + 1 == j) return i % 3 != 2 ? 1 : 0;
    if (i == j + 1) return i % 3 != 0 ? 1 : 0;
    if (i + 3 == j) return 1;
    if (i == j + 3) return 1;
    return 0;

  });
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout << fivePointStencil<double, 3>() << std::endl;
}
{% endhighlight %}

## Result
```
|         -4         1         0         1         0         0         0         0         0 |
|          1        -4         1         0         1         0         0         0         0 |
|          0         1        -4         0         0         1         0         0         0 |
|          1         0         0        -4         1         0         1         0         0 |
|          0         1         0         1        -4         1         0         1         0 |
|          0         0         1         0         1        -4         0         0         1 |
|          0         0         0         1         0         0        -4         1         0 |
|          0         0         0         0         1         0         1        -4         1 |
|          0         0         0         0         0         1         0         1        -4 |
```

**Last Modification date:** 27 February 2018
