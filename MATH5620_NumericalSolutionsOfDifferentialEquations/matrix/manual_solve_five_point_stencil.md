---
title: 5-Point Stencil
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Five Point Stencil

**Routine Name:** solveFivePointStencil

**Author:** Philip Nelson

**Language:** C++

## Description

`fivePointStencil` solves Poisson's Equation for an arbitrary mesh size and forcing function

## Input

`solveFivePointStencil(T a, T b, F f)` requires:

* `T a` - the start boundary
* `T b` - the end boundary
* `F f` - the forcing function
* `T` - template param for the type
* `N` - template param for the dimension of the mesh

## Output

An array with the approximated solution

## Code
{% highlight c++ %}
template <typename T, std::size_t N, typename F>
auto solveFivePointStencil(T a, T b, F f)
{
  auto mesh = generateMesh<double, N>(a, b);
  auto bv = initMeshB(mesh, f);
  auto stencil = fivePointStencil<double, N-2>();
  auto res = stencil.solveLinearSystemLU(bv);
  return res;
}
{% endhighlight %}

solveFivePointStencil relies on:
[fivePointStencil](./manual_gen_five_point_stencil)|
[generateMesh](./manual_gen_mesh)|
[initMeshB](./manual_init_b)|
[solveLinearSystemLU](./manual_linear_solve_lu)|

## Example
{% highlight c++ %}
int main()
{
  auto answer = solveFivePointStencil<double, 5>(0.0, 1.0, sin);
  auto finalMat = arrayToMat(answer);
  std::cout << "Answer in Matrix Form\n" << finalMat << std::endl;
}
{% endhighlight %}

## Result
```
mesh
|     0.0625     0.125     0.188 |
|      0.125      0.25     0.375 |
|      0.188     0.375     0.562 |


stencil
|         -4         1         0         1         0         0         0         0         0 |
|          1        -4         1         0         1         0         0         0         0 |
|          0         1        -4         0         0         1         0         0         0 |
|          1         0         0        -4         1         0         1         0         0 |
|          0         1         0         1        -4         1         0         1         0 |
|          0         0         1         0         1        -4         0         0         1 |
|          0         0         0         1         0         0        -4         1         0 |
|          0         0         0         0         1         0         1        -4         1 |
|          0         0         0         0         0         1         0         1        -4 |


bv
[     0.0625     0.125     0.186     0.125     0.247     0.366     0.186     0.366     0.533 ]


result
[     -0.097    -0.163    -0.154    -0.163    -0.276    -0.266    -0.154    -0.266    -0.266 ]

Answer in Matrix Form
|     -0.097    -0.163    -0.154 |
|     -0.163    -0.276    -0.266 |
|     -0.154    -0.266    -0.266 |
```

**Last Modification date:** 28 February 2018
