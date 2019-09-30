---
title: Cross Product
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Inner Product Software Manual

**Routine Name:** cross_product

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorOps.out** that can be executed.

**Description/Purpose:** The routine calculates the cross product of two vectors. In mathematics and vector algebra, the cross product or vector product is a binary operation on two vectors in three-dimensional space. Given two linearly independent vectors `a` and `b`, the `cross_product(a, b)` is a vector that is perpendicular to both `a` and `b` and thus normal to the plane containing them.

**Input:** The routine takes two inputs which are two vectors.

```
@tparam T Type of the elements in the first vector
@tparam U Type of the elements in the second vector
@tparam R Type of the elements in the result vector
@param a  The first vector
@param b  The second vector
```

**Output:** The routine returns the resulting vector of the cross product of the two input vectors

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> a = {1.1, 2.3, 3.5};
  std::vector<double> b = {4.2, 5.4, 6.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';
  std::cout << "a x b\t" << cross_product(a, b) << "\n";
}
```

**Output** from the lines above
```
a     [        1.1       2.3       3.5 ]

b     [        4.2       5.4       6.6 ]

a x b [      -3.72      7.44     -3.72 ]
```

_explanation of output_:

The first two lines display two vectors `a` and `b`.

The third line is the result of `a x b`.

**Implementation/Code:** The following is the code for cross_product

This code implements the [cross product](https://en.wikipedia.org/wiki/Cross_product) in 3 dimensions. It ensures the sizes of the vectors are the same then returns a vector with the cross product.

``` cpp
template <typename T, typename U, typename R = decltype(T() * U())>
std::vector<R> cross_product(std::vector<T> const& a, std::vector<U> const& b)
{
  // check sizes are 3
  if (a.size() != 3 || b.size() != 3)
  {
    std::cerr << "ERROR: bad size in vector cross product\n";
    exit(EXIT_FAILURE);
  }

  // return the cross product
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
```

**Last Modified:** September 2018
