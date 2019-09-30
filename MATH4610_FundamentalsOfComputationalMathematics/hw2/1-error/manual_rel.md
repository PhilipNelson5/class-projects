---
title: Relative Error
math: true
layout: default
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Relative Error Software Manual

**Routine Name:** relative_error

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./error.out** that can be executed.

**Description/Purpose:** This routine will compute the relative error of an approximation.

Relative error \\(= \eta := \frac{\epsilon}{\lvert v \rvert}, \epsilon := \Big \lvert v - v_{\text{approx}} \Big \rvert \\)

**Input:** The two inputs are the approximate value and the accurate value

```
@tparam T     The type of approx and value
@param approx The approximated value
@param value  The accurate value
```

_Note:_ `approx` and `value` must be the same type.

**Output:** The relative error of type `T`

**Usage/Example:**

``` c++
int main()
{
  auto value = M_PI;
  auto approx = 3.1416;

  std::cout << std::setprecision(15) << "Real Value:  " << value
            << "\nApproximate: " << approx << std::endl
            << std::endl;

  std::cout << "Relative: " << relative_error(approx, value) << std::endl;
}
```

**Output** from the lines above
```
Real Value:  3.14159265358979
Approximate: 3.1416

Relative Error: 2.33843499679617e-06
```

_explanation of output_:
The first two lines show the real/accurate value and the approximate value.
The relative error is computed and displayed.

**Implementation/Code:** The following is the code for `relative_error`

``` c++
template <typename T>
inline T relative_error(const T approx, const T value)
{
  return std::abs((value - approx) / value);
}
```

**Last Modified:** September 2018
