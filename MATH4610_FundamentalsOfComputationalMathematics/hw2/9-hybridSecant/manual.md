---
title: Root Finder Hybrid Secant
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Root Finder Hybrid Secant Software Manual

**Routine Name:** root_finder_hybrid_secant

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./hybridSecant.out** that can be executed.

**Description/Purpose:** This routine will find the roof of a function \\(f\\) on the interval from \\([a, b]\\) using a hybrid method composed of the secant method and the bisection method. Hybrid secant method takes advantage of the bisection method to reduce the interval by ~one order of magnitude, then test the secant method for convergent behavior. If the secant method stays bounded, then the root is found with the secant method. If the secant method leaves the interval, then bisection is used again to reduce the interval.

**Input:** There are four needed inputs, a function, the lower bound of the interval, the upper bound of the interval, and the tolerance.

```
@tparam T       The type of x0 and tolerance
@tparam F       A function of type T(T)
@param a        The lower bound of the interval
@param b        The upper bound of the interval
@param tol      The Tolerance
@param MAX_ITER The maximum iterations
```

**Output:** This routine returns the roof of the given function, \\(f\\) on the interval \\([a, b]\\).

**Usage/Example:**

``` c++
int main()
{
  auto f = [](double x) { return x * x - 3; };

  auto g = [](double x) { return sin(M_PI * x); };

  auto root = root_finder_hybrid_secant(f, 1.0, 10.0, 1e-10);
  if (root.has_value())
    std::cout << std::setprecision(15) << root.value() << std::endl;
  else
    std::cout << "error\n";

  root = root_finder_hybrid_secant(g, 4.1, 5.9, 1e-10);
  if (root.has_value())
    std::cout << std::setprecision(15) << root.value() << std::endl;
  else
    std::cout << "error\n";
}
```

**Output** from the lines above
```
1.73205080756886

5
```

_explanation of output_:

The first line is the root of \\(f(x)\\) on the interval \\((1, 10)\\).

The second line is the root if \\(g(x)\\) on the interval \\((4.1, 5.9)\\)

**Implementation/Code:** The following is the code for root_finder_hybrid_secant

_note_: `bisection_n` is the [bisection method](../5-bisection/manual.md) where the last argument determines the number of iterations.

The code begins by ensuring \\(a < b\\) and if this is not the case, \\(a\\) and \\(b\\) are swapped. Since these define the interval on which we will be looking for a root, we want \\(a < b\\). \\(f\\) is the function whose roots we are interested in finding so we first determine the value of \\(f\\) on the boundaries. If \\(f(a)\\) and \\(f(b)\\) do not have different signs, then by the intermediate value theorem, a zero does not exist between them. Next, we check if \\(f(a)\\) or \\(f(b)\\) is zero because then we have accomplished our goal of finding a zero so we simply return it. Finally we really begin the hybrid secant method. We will use four iterations of bisection method in order to reduce the interval by approximately one order of magnitude. Then we'll try secant method with the maximum iterations set to 1. This will only do one iteration of secant method and we will test the output to see if it stays within our interval \\([a, b]\\). If it stays within the boundary, then it will converge so the loop ends and we use secant method to finish finding the root. If it doesn't stay in the boundary then we do another four iterations of bisection to reduce the interval again and repeat.

``` c++
template <typename T, typename F>
std::optional<T> root_finder_hybrid_secant(F f, T a, T b, T tol)
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

  T x0;
  do
  {
    // four iterations of bisection to reduce interval
    // by ~one order of magnitude
    std::tie(a, b) = bisection_n(f, a, b, f(a), 4);

    // check secant method for convergent behavior
    x0 = root_finder_secant(f, a, b, tol, 1);

  } while (a > x0 || x0 > b);

  return root_finder_secant(f, a, b, tol);
}
```

**Last Modified:** September 2018
