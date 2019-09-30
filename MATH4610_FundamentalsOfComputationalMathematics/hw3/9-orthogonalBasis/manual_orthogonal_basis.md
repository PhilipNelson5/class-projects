---
title: Orthogonal Basis
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
#Orthogonal Basis Software Manual

**Routine Name:** orthogonal_basis

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./orthogonalBasis.out** that can be executed.

**Description/Purpose:** An orthogonal basis for an inner product space V is a basis for V whose vectors are mutually orthogonal. If the vectors of an orthogonal basis are normalized, the resulting basis is an orthonormal basis.

**Input:**

The code takes two parameters, the vectors a and b, from which an orthogonal basis is formed.

```
@tparam T Type of the elements of vector a and b
@param a  The first vector
@param b  The second vector
```

**Output:**

The code returns a tuple of vectors which represent the orthogonal basis for the vector space.

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> v1 = {2, 5};
  std::vector<double> v2 = {6, 5};

  std::cout << "v1 " << v1 << '\n' << "v2 " << v2 << '\n';

  auto [u1, u2] = orthogonal_basis(v1, v2);
  std::cout << "u1 " << u1 << '\n' << "u2 " << u2 << std::endl;
  std::cout << "u1 · u2 " << inner_product(u1, u2) << std::endl;

  std::cout << "\n------------------------------\n" << std::endl;

  v1 = {1, 4};
  v2 = {-5, 2};

  std::cout << "v1 " << v1 << '\n' << "v2 " << v2 << '\n';

  std::tie(u1, u2) = orthogonal_basis(v1, v2);
  std::cout << "u1 " << u1 << '\n' << "u2 " << u2 << std::endl;
  std::cout << "u1 · u2 " << inner_product(u1, u2) << std::endl;
}
```

**Output** from the lines above
```
v1 [          2         5 ]

v2 [          6         5 ]

u1 [      -0.64     0.768 ]

u2 [      0.768      0.64 ]

u1 · u2 -5.55e-17

------------------------------

v1 [          1         4 ]

v2 [         -5         2 ]

u1 [      0.371     0.928 ]

u2 [     -0.928     0.371 ]

u1 · u2 0
```

_explanation of output_:

Two examples are shown. The initial vectors, v1 and v2, are displayed and then the resulting orthogonal vectors, u1 and u2. We know that they are orthogonal because their dot product is zero, or near zero due to floating point error.

**Implementation/Code:** The following is the code for orthogonal_basis

``` cpp
template <typename T>
std::tuple<std::vector<T>, std::vector<T>> orthogonal_basis(std::vector<T> a,
                                                            std::vector<T> b)
{
  // check that the vectors are in R2
  if (a.size() != 2 || b.size != 2)
  {
    std::cerr << "[ERROR] vectors not in R2 in orthogonal_basis" << std::endl;
  }

  // normalize v2
  auto bn = b / p_norm(b, 2);

  // the projection of v1 onto v2
  std::vector<T> proj = inner_product(a, bn) * bn;

  // find the vector orthogonal to v2 which points to v1
  std::vector<T> ortho = a - proj;

  // normalize
  std::vector<T> u1 = ortho / p_norm(ortho, 2);
  std::vector<T> u2 = bn;

  // return the two orthogonal vectors
  return {u1, u2};
}
```

**Last Modified:** October 2018
