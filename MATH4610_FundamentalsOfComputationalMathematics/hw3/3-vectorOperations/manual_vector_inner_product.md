---
title: Inner Product
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Inner Product Software Manual
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

**Routine Name:** inner_product

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorOps.out** that can be executed.

**Description/Purpose:** The routine calculates the inner product of two vectors also commonly known as the dot product. In mathematics, the dot product is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors) and returns a single number.

**Input:** The routine takes two inputs which are two vectors.

```
@tparam T Type of the elements in the first vector
@tparam U Type of the elements in the second vector
@tparam R Type of the elements in the result vector
@param a  The first vector
@param b  The second vector
```

**Output:** The routine returns the resulting dot product of the two input vectors

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> a = {1.1, 2.3, 3.5};
  std::vector<double> b = {4.2, 5.4, 6.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';
  std::cout << "a · b\t" << inner_product(a, b) << "\n\n";
}
```

**Output** from the lines above
```
a     [        1.1       2.3       3.5 ]

b     [        4.2       5.4       6.6 ]

a · b 40.1
```

_explanation of output_:

The first two lines display two vectors `a` and `b`.

The third line is the result of `a·b`.

**Implementation/Code:** The following is the code for inner_product

``` cpp
template <typename T, typename U, typename R = decltype(T() * U())>
R inner_product(std::vector<T> const& a, std::vector<U> const& b)
{
  // check the sizes are the same
  if (a.size() != b.size())
  {
    std::cerr << "ERROR: bad size in vector inner product\n";
    exit(EXIT_FAILURE);
  }

  // initalize the result of the inner product
  R product = 0.0;

  // multiply the vectors element wise and add to the result
  for (auto i = 0u; i < a.size(); ++i)
  {
    product += a[i] * b[i];
  }

  // return the inner product
  return product;
}
```

**Last Modified:** October 2018
