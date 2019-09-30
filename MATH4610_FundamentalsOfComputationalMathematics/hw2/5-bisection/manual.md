---
title: Root Finder Bisection
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Root Finder Bisection Software Manual

**Routine Name:** root_finder_bisection

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./bisection.out** that can be executed.

**Description/Purpose:** This routine will find the root of a function \\(f\\) on the interval \\((a, b)\\) using the bisection method of root finding.

**Input:** There are four needed inputs, a function, the beginning of the interval, the end of the interval and the tolerance.

```
@tparam T  Type of a and b, determines the return type
@tparam F  A function of type T(T)
@param f   The function to find roots of
@param a   The beginning of the interval
@param b   The end of the interval
@param tol The tolerance
```

**Output:** This routine returns an optional `T` which contains the root of the function. If the parameters of the function are not valid, `a > b` or `f(a) * f(b) >= 0` or `tol <= 0`, then an empty optional is returned.

**Usage/Example:**

The following is an example using two functions, \\(f(x) = x^2 - 3\\) and \\(g(x) = sin(\pi \cdot x)\\).

``` c++
int main()
{
  auto f = [](double x) { return x * x - 3; };
  auto g = [](double x) { return sin(M_PI*x); };

  auto root = root_finder_bisection(f, 0.0, 5.5, 1e-100);
  if (root.has_value())
  {
    std::cout << std::setprecision(20) << root.value() << '\n';
  }
  else
  {
    std::cout << "no roots on specified interval\n";
  }

  root = root_finder_bisection(g, 4.5, 5.5, 1e-100);
  if (root.has_value())
  {
    std::cout << std::setprecision(20) << root.value() << '\n';
  }
  else
  {
    std::cout << "no roots on specified interval\n";
  }
}
```

**Output** from the lines above
```
1.7320508075688771932

5
```

_explanation of output_:
The first line is the root of \\(f(x)\\) on the interval \\((0, 5)\\).

The second line is the root if \\(g(x)\\) on the interval \\((4.5, 5.5)\\)

**Implementation/Code:** The following is the code for `root_finder_bisection`

``` c++
template <typename T, typename F>
std::optional<T> root_finder_bisection(F f, T a, T b, T tol)
{
  if (a > b)
  {
    std::swap(a, b);
  }

  auto fa = f(a);
  auto fb = f(b);

  if (fa * fb > 0 || tol <= 0)
  {
    return {};
  }

  if (fa == 0) return a;
  if (fb == 0) return b;

  T p, fp;
  auto n = std::ceil(log2((b - a) / (2 * tol)));

  for (auto i = 0; i < n; ++i)
  {
    p = (a + b) / 2;
    fp = f(p);

    if (fa * fp < 0) // root on interval [a, p]
    {
      b = p;
    }
    else // root in interval [p, b]
    {
      a = p;
      fa = fp;
    }
  }

  return (a + b) / 2;
}
```

**Last Modified:** September 2018
