---
title: Building Library
layout: default
---
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Building Library Software Manual

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable that is linked to libmaceps.a **./maceps .out** that can be executed.

**Description/Purpose:** This code shows the capacity to compile and link to an external library.

**Input:** There are no inputs needed in this case.

**Output:** This code demonstrates `smaceps` and `dmaceps` for calculating the single and double machine precision for any computer.

**Usage/Example:**

``` c++
#include <iostream>
#include "maceps.hpp"

int main()
{
  auto [sprec, seps] = smaceps();
  auto [dprec, deps] = dmaceps();

  std::cout << "single\t" << sprec << '\t' << seps << '\n';
  std::cout << "double\t" << dprec << '\t' << deps << '\n';
}
```

**Output** from the lines above
```
single	24	1.19209e-07
double	53	2.22045e-16
```

**Implementation/Code:** The following are the steps taken to compile the library:

```
# g++ -std=c++17 -O3 -c maceps.cpp
# ar crv libmaceps.a maceps.o

# make
g++ -std=c++17 -O3 main.cpp -L./ -lmaceps -o maceps.out
```

**Last Modified:** September 2018
