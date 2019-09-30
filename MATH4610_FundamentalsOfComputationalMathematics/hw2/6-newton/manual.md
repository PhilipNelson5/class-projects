---
title: Root Finder Newton's
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Root Finder Newton's Software Manual

**Routine Name:** root_finder_newton

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./newton.out** that can be executed.

**Description/Purpose:** This routine will find the root of a function \\(f\\) starting at \\(x\\) using newton's method of root finding.

**Input:** there are five needed inputs, a function, it's derivative, the starting point, the tolerance, and a maximum number of iterations.

```
@tparam T       The type of x0 and tolerance
@tparam F       A function of type T(T)
@tparam Fprime  A function of type T(T)
@param x0       The starting point
@param tol      The Tolerance
@param MAX_ITER The maximum iterations
```

**Output:** This routine returns the root of the given function \\(f\\).

**Usage/Example:**

The following is an example using two functions, \\(f(x) = x^2 - 3\\) and \\(g(x) = sin(\pi \cdot x)\\).
``` c++
int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto fprime = [](double x) { return 2 * x; };

  auto g = [](double x) { return sin(M_PI * x); };
  auto gprime = [](double x) { return M_PI * cos(M_PI * x); };

  auto root = root_finder_newton(f, fprime, 3.0, 1e-100, 100);
  std::cout << std::setprecision(15) << root << std::endl;

  root = root_finder_newton(g, gprime, 4.75, 1e-100, 100);
  std::cout << std::setprecision(15) << root << std::endl;
}
```

**Output** from the lines above
```
1.73205080756888

5
```

_explanation of output_:
The first line is the root of \\(f(x)\\) starting at \\(3\\).

The second line is the root if \\(g(x)\\) starting at \\(4.75\\)

**Implementation/Code:** The following is the code for root_finder_newton

``` c++
template <typename T, typename F, typename Fprime>
T root_finder_newton(F f, Fprime fprime, T x0, T tol, const int MAX_ITER = 100)
{
  T x1;

  for (auto i = 0; i < MAX_ITER; ++i)
  {
    x1 = x0 - f(x0) / fprime(x0);
    if (std::abs(x1 - x0) < tol * std::abs(x1))
    {
      break;
    }
    x0 = x1;
  }

  return x1;
}
```

**Last Modified:** September 2018
