---
title: Nine Point Stencil
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Nine Point Stencil

**Routine Name:** ninePointStencil

**Author:** Philip Nelson

**Language:** C++

## Description

`ninePointStencil` returns an \\(N^2\\)x\\(N^2\\) matrix for the nine point stencil for 2D finite difference method

## Input

`Matrix<T, N * N, N * N> ninePointStencil()` takes no input but requires the size and template parameters \\(N\\) and \\(T\\)

## Output

A \\(N^2\\)x\\(N^2\\) matrix with the stencil

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
Matrix<T, N * N, N * N> ninePointStencil()
{
  return Matrix<T, N * N, N * N>([](int i, int j) {
    if (i == j) return -20;
    if (i + 1 == j) return i % 3 != 2 ? 4 : 0;
    if (i == j + 1) return i % 3 != 0 ? 4 : 0;
    if (i + 2 == j) return i % 3 != 0 ? 1 : 0;
    if (i == j + 2) return i % 3 != 2 ? 1 : 0;
    ;
    if (i + 3 == j) return 4;
    if (i == j + 3) return 4;
    if (i + 4 == j) return i % 3 != 2 ? 1 : 0;
    if (i == j + 4) return i % 3 != 0 ? 1 : 0;
    ;
  });
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout << ninePointStencil<double, 3>() << std::endl;
}
{% endhighlight %}

## Result
```
|        -20         4         0         4         1         0         0         0         0 |
|          4       -20         4         1         4         1         0         0         0 |
|          0         4       -20         0         1         4         0         0         0 |
|          4         1         0       -20         4         0         4         1         0 |
|          1         4         1         4       -20         4         1         4         1 |
|          0         1         4         0         4       -20         0         1         4 |
|          0         0         0         4         1         0       -20         4         0 |
|          0         0         0         1         4         1         4       -20         4 |
|          0         0         0         0         1         4         0         4       -20 |
```

**Last Modification date:** 27 February 2018
