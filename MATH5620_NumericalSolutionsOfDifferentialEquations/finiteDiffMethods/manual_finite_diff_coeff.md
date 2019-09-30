---
title: Finite Difference Coefficients
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Finite Difference Coefficients

**Routine Name:** centralFinDiffCoeff

**Author:** Philip Nelson

**Language:** C++

## Description

`centralFinDiffCoeff` returns a vector of the coefficients for finite difference approximations of an arbitrary order of accuracy for a given derivative. This routine uses binomial coefficients to calculate the coefficients with the formula below:

```
$ make
$ ./finDiffCoeff.out
```

This will compile and run the driver program.

## Input

`centralFinDiffCoeff()` takes no parameters, however it requires the type, order and accuracy as template parameters.

* `T` - the type you want the coefficients in
* `ord` - the order of the derivative
* `acc` - the accuracy of the approximation

## Output

`centralFinDiffCoeff` returns a vector of coefficients.

## Code
{% highlight c++ %}
template <typename T, std::size_t ord, std::size_t acc>
auto centralFinDiffCoeff()
{
  constexpr int size = 2.0 * std::floor((ord + 1.0) / 2.0) - 1.0 + acc;
  constexpr int P = (size - 1.0) / 2.0;

  Matrix<double, size, size> mat;
  for (auto i = 0; i < size; ++i)
  {
    for (auto j = 0; j < size ; ++j)
    {
      mat[i][j] = std::pow(-P+j, i);
    }
  }

  std::array<T, size> b;
  b.fill(0.0);
  b[ord] = fact(ord);

  return mat.solveLinearSystemLU(b);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  auto coeffs = centralFinDiffCoeff<double, 1, 4>();

  std::cout << "coefficients of a second order derivative with 4th accuracy\n\n";
  std::cout << coeffs << std::endl;

  return EXIT_SUCCESS;
}
{% endhighlight %}

## Result
```
coefficients of a second order derivative with 4th accuracy

[    -0.0833      1.33      -2.5      1.33   -0.0833 ]

```

**Last Modification date:** 11 January 2018
