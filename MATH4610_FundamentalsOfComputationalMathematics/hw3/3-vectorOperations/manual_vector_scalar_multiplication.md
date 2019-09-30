---
title: Vector Scalar Multiplication
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Vector Scalar Multiplication Software Manual

**Routine Name:** Vector Scalar Multiplication

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorOps.out** that can be executed.

**Description/Purpose:** This routine overloads the `*` operator in c++ allowing multiplication of a scalar and a vector

**Input:** The operator requires two operands, `s` and `a`, where `a` is a `std::vector<T>` and `s` is a scalar value

```
@tparam T Type of the elements in the first vector
@tparam U Type of the scalar
@tparam R Type of the elements in the result vector
@param s  The scalar
@param a  The first vector
```

**Output:** The result of multiplying the scalar `s` and vector `a`

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> a = {1.1, 2.3, 3.5};
  std::cout << "a\t" << a << '\n';
  std::cout << "7.9 * a\t" << 7.9 * a << '\n';
}
```

**Output** from the lines above
```
a       [        1.1       2.3       3.5 ]

7.9 * a [       8.69      18.2      27.7 ]
```

_explanation of output_:

The first line displays the vector `a`.

The second line is the result of `7.9*a`.

**Implementation/Code:** The following is the code for Vector Scalar Multiplication

In order to make vector scalar multiplication intuitive to use, the routine is implemented as an overload of the `operator*`.

In order to make vector scalar multiplication more generalized, this routine takes advantage of c++ templates allowing the multiplication of a scalar of one type by a vector of another as long as the `*` operator is defined for the two types. No branching structure required.

This code uses [std::transform](https://en.cppreference.com/w/cpp/algorithm/transform). It transforms each element of the vector `a` and it puts the result of the transformation into the result vector. The transformation is is a unary operation that takes the element of the vector and multiplies it by the scalar `s`. The transformed result vector is then returned.


``` cpp
  template <typename T, typename U, typename R = decltype(T() * U())>
  std::vector<R> operator*(U const s, std::vector<T> const& a)
{
  std::vector<R> result(a.size());
  std::transform(
    std::begin(a), std::end(a), std::begin(result), [s](T e) { return e * s; });
  return result;
}
```

**Last Modified:** October 2018
