---
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/class-projects/MATH5620_NumericalSolutionsOfDifferentialEquations/SoftwareManual"> Table of Contents </a>
# Error

**Routine Name:** solcc

**Author:** Philip Nelson

**Language:** C++

## Description

solcc computes the solution to the below **s**econd-**o**rder, **l**inear, **c**onstant-**c**oefficient equation at a time _t_.

\\[ ay^{\prime \prime} + by^{\prime} + cy = f(t) \\]

```
$ make
$ ./error.out
```

This will compile and run the driver program.

## Input absoluteError

`std::optional<std::complex<N>> solcc(N y0, N v0, A a, B b, C c, T t)`  requires:
* N y0 - initial y
* N v0 - initial v
* A a - coefficent on y\`\`
* B b - coefficent on y\`\`
* C c - coefficent on y\`\`
* T t - time

## Output

`solcc` returns a `std::optional<std::complex<N>>` with the solution.

## Code
{% highlight c++ %}
template <typename N, typename A, typename B, typename C, typename T>
std::optional<std::complex<N>> solcc(N y0, N v0, A a, B b, C c, T t)
{
  // roots from the quadratic formula
  std::complex<N> const radical = (b * b) - (4.0 * a * c);
  auto const r1 = (-b + sqrt(radical)) / (2.0 * a);
  auto const r2 = (-b - sqrt(radical)) / (2.0 * a);

  // There is no solution if the roots are the same
  if (r1 == r2) return {};

  // calculate c1 and c2
  auto const c1 = (v0 - (r2 * y0)) / (r1 - r2);
  auto const c2 = ((r1 * y0) - v0) / (r1 - r2);

  // return the solution
  return c1 * exp(r1 * t) + c2 * exp(r2 * t);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  auto result = solcc(1.0, 0.0, 2.0, 3.0, -2.0, 4.0).value();
  std::cout << result << std::endl;
}
{% endhighlight %}

## Result
```
(5.91131,0)
```
**Last Modification date:** 22 January 2018
