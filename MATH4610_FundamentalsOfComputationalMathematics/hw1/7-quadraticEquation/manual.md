---
title: Quadratic Equation
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Quadratic Equation Software Manual
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

**Routine Name:** `quadratic_equation`

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./quadratic.out** that can be executed.

**Description/Purpose:** This routine computes the solutions to the second order polynomial, \\(ax^2 + bx + c\\) and returns both solutions if they exist.

**Input:** There are three inputs required which represent \\(a,b,c\\) of the second order polynomial.

**Output:** This routine returns an optional array of two numbers matching the type of the input parameters. If the solutions are real, they are returned in an optional array. If the solutions are imaginary, the optional is returned with no values. Thus, the caller must check if the result has any value.

**Usage/Example:**

``` c++
#include "quadraticEquation.hpp"
#include <iostream>

int main()
{
  auto res = quadratic_equation(2.0, 9.0, -5.0);
  if (res)
  {
    auto [r1, r2] = res.value();
    std::cout << "( " << r1 << " , " << r2 << " )\n";
  }
  else
  {
    std::cout << "imaginary roots\n";
  }

  res = quadratic_equation(2.0, 3.0, 5.0);
  if (res)
  {
    auto [r1, r2] = res.value();
    std::cout << r1 << ' ' << r2 << '\n';
  }
  else
  {
    std::cout << "imaginary roots\n";
  }
  return EXIT_SUCCESS;
}
```

**Output** from the lines above
```
( 0.5,-5 )
imaginary roots
```

The first line contains the roots from the equation \\(2x^2 + 9x - 5\\). The second equation \\(2x^2 + 3x + 5\\) has imaginary roots as is reported in the second line.

**Implementation/Code:** The following is the code for `quadratic_equation`

``` c++
#include <array>
#include <cmath>
#include <optional>

template <typename T>
std::optional<std::array<T, 2>> quadratic_equation(T a, T b, T c)
{
  const auto descrim = (b * b) - (4.0 * a * c);
  if (descrim > 0)
  {
    const T r1 = (-b + sqrt(descrim)) / (2.0 * a);
    const T r2 = (-b - sqrt(descrim)) / (2.0 * a);

    return std::array {r1, r2};
  }
  return {};
}
```

**Last Modified:** September 2018
