---
title: Vector Relative Error
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Relative Vector Relative Error Software Manual

**Routine Name:** relative_error

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorError.out** that can be executed.

**Description/Purpose:** This routine will compute the relative error of an approximate vector.

Relative error \\(= \Big \lvert ||v|| - ||v_{\text{approx}}|| \Big \rvert \\)

**Input:** The three inputs are the approximate vector, the accurate vector, and a norm function.

```
@tparam T     The type of the elements in approx and value
@tparam F     A function that takes a std::vector<T> and returns a T
@param approx The approximated vector
@param value  The accurate vector
@param norm   A function that takes a vector and returns a T
```

_Note:_ `approx` and `value` must be the same type.

**Output:** The relative error of type `T`

**Usage/Example:**

``` c++
int main()
{
  std::vector<double> a = {1.1, 2.22, 3.5};
  std::vector<double> b = {1.2, 2.23, 3.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';

  std::cout << "relative error of a and b\n"
            << relative_error(
                 a, b, std::bind(p_norm<double, int>, std::placeholders::_1, 2))
            << std::endl;
}
```

**Output** from the lines above
```
a	[        1.1      2.22       3.5 ]

b	[        1.2      2.23       3.6 ]

relative error of a and b
0.0257
```

_explanation of output_:

The first two lines show the real/accurate vector and the approximate vector.
The relative error is computed and displayed.

**Implementation/Code:** The following is the code for `relative_error`

The implementation for `relative_error` takes two vectors and a norm function. This way the error can be caculated using whatever norm you desire. The norm function must be of the type `T(std::vector<T>)` which means that it must take a `std::vector<T>` and return a single `T`. It then follows the standard definition for norm-wise relative error.

``` c++
template <typename T, typename F>
inline T relative_error(std::vector<T> const& approx,
                        std::vector<T> const& value,
                        F norm)
{
  auto valNorm = norm(value);
  return std::abs((norm(approx) - valNorm) / valNorm);
}
```

**Last Modified:** October 2018
