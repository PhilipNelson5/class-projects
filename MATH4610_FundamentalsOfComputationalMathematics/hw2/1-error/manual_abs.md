---
title: Absolute Error
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Absolute Error Software Manual

**Routine Name:** absolute_error

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./error.out** that can be executed.

**Description/Purpose:** This routine will compute the absolute error of an approximation.

absolute error \\(= \epsilon := \Big \lvert v - v_{\text{approx}} \Big \rvert \\)

**Input:** The two inputs are the approximate value and the accurate value

```
@tparam T     The type of approx and value
@param approx The approximated value
@param value  The accurate value
```

_Note:_ `approx` and `value` must be the same type.

**Output:** The absolute error of type `T`

**Usage/Example:**

``` c++
int main()
{
  auto value = M_PI;
  auto approx = 3.1416;

  std::cout << std::setprecision(15) << "Real Value:  " << value
            << "\nApproximate: " << approx << std::endl
            << std::endl;

  std::cout << "Absolute: " << absoluteError(approx, value) << std::endl;
}
```

**Output** from the lines above
```
Real Value:  3.14159265358979
Approximate: 3.1416

Absolute Error: 7.34641020683213e-06
```

_explanation of output_:
The first two lines show the real/accurate value and the approximate value.
The absolute error is computed and displayed.

**Implementation/Code:** The following is the code for `absolute_error`

``` c++
template <typename T>
inline T absolute_error(const T approx, const T value)
{
  return std::abs(value - approx);
}
```

**Last Modified:** September 2018
