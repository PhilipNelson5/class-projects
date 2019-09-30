---
title: Root Finder Fixed Point Iteration
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Root Finder Fixed Point Iteration Software Manual

**Routine Name:** root_finder_fixed_point_iteration

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./fixedIter.out** that can be executed.

**Description/Purpose:** This routine will find the root of a function \\(f\\) starting with the initial guess \\(x0\\) using fixed point iteration.

**Input:** There are four needed inputs: a function \\(g(x)\\), an initial guess, the tolerance and a maximum number of iterations.

```
@tparam T       The type of x0 and tol
@tparam G       A function of type T(T)
@param x0       The initial guess
@param tol      The tolerance
@param MAX_ITER The maximum number of iterations
```

**Output:** This routine returns the root of type `T` of the original function \\(f(x)\\).

**Usage/Example:**

Examples of functional iteration using \\(f_1(x) = x^2 - 3\\) where \\(g_1(x) = x - \frac{x^2 - 3}{10}\\)

and \\(f_2(x) = \sin(\pi\cdot x)\\) where \\(g_2(x) = x - \frac{\sin(\pi\cdot x)}{2}\\)

``` c++
int main()
{
  auto g1 = [](double x) { return x - (x * x - 3) / 10; };
  auto g2 = [](double x) { return x - sin(M_PI * x) / 2; };

  auto approx1 = root_finder_fixed_point_iteration(g1, 5.3, 1.0e-5);
  std::cout << approx1 << '\n';

  auto approx2 = root_finder_fixed_point_iteration(g2, 5.8, 1.0e-5);
  std::cout << approx2 << '\n';
}
```

**Output** from the lines above
```
1.73207

6
```

_explanation of output_:

The first line is a root of \\(f_1(x)\\)

The second line is a root of \\(f_2(x)\\)

**Implementation/Code:** The following is the code for root_finder_fixed_point_iteration

``` c++
template <typename T, typename G>
T root_finder_fixed_point_iteration(G g, T x0, T tol, const int MAX_ITER = 100)
{
  T x1;
  for(auto i = 0.0; i < MAX_ITER; ++i)
  {
    x1 = g(x0);
    if(std::abs(x1-x0) < tol)
      return x1;
    x0 = x1;
  }
  return x0;
}
```

**Last Modified:** September 2018
