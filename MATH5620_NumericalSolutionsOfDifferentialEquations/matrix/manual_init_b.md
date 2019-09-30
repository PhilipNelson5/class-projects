---
title: Initialize B for Mesh
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Initialize B for Mesh

**Routine Name:** initMeshB

**Author:** Philip Nelson

**Language:** C++

## Description

`initMeshB` initializes the `b` matrix used for the 2D finite difference method

## Input

`initMeshB(Matrix<T, N, N>& mesh, F f)` requires:

* `Matrix<T, N, N>` - the `N`x`N` mesh
* `F f` - the forcing function

## Output

The `b` matrix used to solve the 2D finite difference method

## Code
{% highlight c++ %}
template <typename T, typename F, std::size_t N>
std::array<T, N * N> initMeshB(Matrix<T, N, N>& mesh, F f)
{
  std::array<T, N * N> b;
  for (auto i = 0u; i < N; ++i)
  {
    for (auto j = 0u; j < N; ++j)
    {
      b[i * N + j] = f(mesh[i][j]);
    }
  }
  return b;
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  auto mesh = generateMesh<double, 7>(0, 1);
  auto b = initMeshB(mesh, sin);
  std::cout << "b\n" << b << std::endl << std::endl;
}
{% endhighlight %}

## Result
```
b
[     0.0278    0.0555    0.0832     0.111     0.138    0.0555     0.111     0.166      0.22     0.274    0.0832     0.166     0.247     0.327     0.405     0.111      0.22     0.327      0.43     0.527     0.138     0.274     0.405     0.527      0.64 ]
```

**Last Modification date:** <++>10 February 2018
