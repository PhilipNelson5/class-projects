---
title: Root Finder Secant
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Root Finder Secant Method Software Manual
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

**Routine Name:** root_finder_secant

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./secant.out** that can be executed.

**Description/Purpose:** This routine will find the root of a function, \\(f\\), using two initial guesses \\(x0\\) and \\(x1\\) using the secant method of root finding

**Input:** There are five needed inputs, a function, two initial guesses, a tolerance and the maximum number of iterations.

```
@tparam T       The type of x0 and tolerance
@tparam F       A function of type T(T)
@param x0       The first initial guess
@param x1       The second initial guess
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
  auto g = [](double x) { return sin(M_PI * x); };

  auto root = root_finder_secant(f, 4.0, 5.5, 1e-10);
  std::cout << std::setprecision(20) << root << '\n';

  root = root_finder_secant(g, 3.5, 6.5, 1e-10);
  std::cout << std::setprecision(20) << root << '\n';
}
```

**Output** from the lines above
```
1.7320508075688771932

5
```

_explanation of output_:

The first line is the root of \\(f(x)\\) starting with guesses \\(4.0\\) and \\(5.5\\).

The second line is the root of \\(g(x)\\) starting with guesses \\(3.5\\) and \\(6.5\\).

**Implementation/Code:** The following is the code for root_finder_secant

``` c++
template <typename T, typename F>
T root_finder_secant(F f, T x0, T x1, T tol, const int MAX_ITER = 100)
{
  T x2, fx1;
  for (auto i = 0; i < MAX_ITER; ++i)
  {
    fx1 = f(x1);
    x2 = x1 - fx1 * (x1 - x0) / (fx1 - f(x0));
    if (std::abs(x1 - x0) < tol * std::abs(x1))
    {
      return x2;
    }
    x0 = x1;
    x1 = x2;
  }
  return x1;
}
```

**Last Modified:** September 2018
