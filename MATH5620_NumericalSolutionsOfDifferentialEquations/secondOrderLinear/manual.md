---
title: Second Order Linear
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Second Order Linear Constant Coefficents

**Routine Name:** solcc

**Author:** Philip Nelson

**Language:** C++

## Description

`solcc` computes the solution to the **s**econd-**o**rder **l**inear **c**onstant-**c**oefficient equation below at time \\(t\\).

\\[ ay^{\prime \prime} + by^{\prime} + cy = f(t) \\]

```
$ make
$ ./solcc.out
```

This will compile and run the driver program.

## Input

`T solcc(T y0, T v0, T a, T b, T c, T t)`  requires:
* `T y0` - initial condition \\(y(0)\\)
* `T v0` - initial condition \\(y^{\prime}(0)\\)
* `T  a` - coefficent \\(a\\) on \\(y^{\prime\prime}\\)
* `T  b` - coefficent \\(b\\) on \\(y^{\prime}\\)
* `T  c` - coefficent \\(c\\) on \\(y\\)
* `T  t` - time

**Note:** all parameters must be the same type.

## Output

`solcc` returns a `std::optional<std::complex<N>>` with the solution.

## Code
{% highlight c++ %}
template <typename T>
T solcc(T y0, T v0, T a, T b, T c, T t)
{
  // roots from the quadratic formula
  std::complex<T> const sqDiscrim = sqrt((b * b) - (4.0 * a * c));
  auto const r1 = (-b + sqDiscrim) / (2.0 * a);
  auto const r2 = (-b - sqDiscrim) / (2.0 * a);

  if (r1 == r2) // double roots
  {
    // calculate c1 and c2
    auto const c1 = y0;
    auto const c2 = v0 - r1 * y0;

    // return the solution
    return std::real(c1 * exp(r1 * t) + c2 * t * exp(r2 * t));
  }
  else // unique roots
  {
    // calculate c1 and c2
    auto const c1 = (v0 - (r2 * y0)) / (r1 - r2);
    auto const c2 = ((r1 * y0) - v0) / (r1 - r2);

    // return the solution
    return std::real(c1 * exp(r1 * t) + c2 * exp(r2 * t));
  }
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  auto solution = solcc(2.0, 0.0, 3.0, 5.0, -1.0, 3.0);
  std::cout << solution << std::endl;
}
{% endhighlight %}

## Result
```
3.13158
```
**Last Modification date:** 22 January 2018
