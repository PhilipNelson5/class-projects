---
title: 9-Point Stencil
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Nine Point Stencil

**Routine Name:** solveNinePointStencil

**Author:** Philip Nelson

**Language:** C++

## Description

`solveNinePointStencil` solves Poisson's Equation for an arbitrary mesh size and forcing function

## Input

`solveNinePointStencil(T a, T b, F f)` requires:

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
  auto stencil = ninePointStencil<double, N-2>();
  auto res = stencil.solveLinearSystemLU(bv);
  return res;
}
{% endhighlight %}

solveFivePointStencil relies on:
[ninePointStencil](./manual_gen_nine_point_stencil)|
[generateMesh](./manual_gen_mesh)|
[initMeshB](./manual_init_b)|
[solveLinearSystemLU](./manual_linear_solve_lu)|

## Example
{% highlight c++ %}
int main()
{
  auto answer = solveNinePointStencil<double, 5>(0.0, 1.0, sin);
  auto finalMat = arrayToMat(answer);
  std::cout << "Answer in Matrix Form\n" << finalMat << std::endl;
}
{% endhighlight %}

## Result
```
|     0.0625     0.125     0.188 |
|      0.125      0.25     0.375 |
|      0.188     0.375     0.562 |


stencil
|        -20         4         0         4         1         0         0         0         0 |
|          4       -20         4         1         4         1         0         0         0 |
|          0         4       -20         0         1         4         0         0         0 |
|          4         1         0       -20         4         0         4         1         0 |
|          1         4         1         4       -20         4         1         4         1 |
|          0         1         4         0         4       -20         0         1         4 |
|          0         0         0         4         1         0       -20         4         0 |
|          0         0         0         1         4         1         4       -20         4 |
|          0         0         0         0         1         4         0         4       -20 |


bv
[     0.0625     0.125     0.186     0.125     0.247     0.366     0.186     0.366     0.533 ]


result
[    -0.0169   -0.0284   -0.0267   -0.0284   -0.0483   -0.0466   -0.0267   -0.0466   -0.0477 ]

Answer in Matrix Form
|    -0.0169   -0.0284   -0.0267 |
|    -0.0284   -0.0483   -0.0466 |
|    -0.0267   -0.0466   -0.0477 |
```

**Last Modification date:** 28 February 2018
