---
title: generateMesh
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Generate Mesh

**Routine Name:** generateMesh

**Author:** Philip Nelson

**Language:** C++

## Description

`generateMesh(T a, T b)` generates a 2D mesh from `a` to `b` with `N` steps

## Input

`generateMesh(T a, T b)` requires:

* `T a` - the start of the mesh
* `T b` - the end of the mesh
* `N` - the size

## Output

The mesh of \\(i\cdot j\\)

## Code
{% highlight c++ %}
template <typename T, std::size_t N>
Matrix<T, N - 2, N - 2> generateMesh(T a, T b)
{
  auto h = (b - a) / (N - 1);

  return Matrix<T, N - 2, N - 2>(
    [&](int i, int j) { return (a + (i + 1) * h) * (a + (j + 1) * h); });
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  std::cout << "mesh\n" << generateMesh<double, 7>(0, 1) << std::endl;
}
{% endhighlight %}

## Result
```
mesh
|     0.0278    0.0556    0.0833     0.111     0.139 |
|     0.0556     0.111     0.167     0.222     0.278 |
|     0.0833     0.167      0.25     0.333     0.417 |
|      0.111     0.222     0.333     0.444     0.556 |
|      0.139     0.278     0.417     0.556     0.694 |
```

**Last Modification date:** 27 February 2018
