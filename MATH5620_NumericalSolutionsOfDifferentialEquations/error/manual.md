---
title: Error
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/class-projects/MATH5620_NumericalSolutionsOfDifferentialEquations/SoftwareManual"> Table of Contents </a>
# Error

**Routine Names:** [absoluteError](#input-absoluteerror) and [relativeError](#input-relativeerror)

**Author:** Philip Nelson

**Language:** C++

## Description

`absoluteError` calculates the _absolute_ error given an approximation and a real value.

`relativeError` calculates the _relative_ error given an approximation and a real value.

A driver program and make file are provided.

```
$ make
$ ./error.out
```

This will compile and run the driver program.

## Input absoluteError

`absoluteError<T>(T approx, T value)` requires:
* `T approx` - the approximated value
* `T value` - the real value

absolute error \\(= \epsilon := \Big \lvert v - v_{\text{approx}} \Big \rvert \\)


**Note:** `approx` and `value` must be the same type.

## Output

`absoluteError` returns a `T` with the absolute error.

## Code
{% highlight c++ %}
template <typename T>
inline T absoluteError(const T approx, const T value)
{
  return std::abs(value - approx);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  double approx = 3.2;
  double value = 3.14159;

  std::cout << "Approximate: " << approx
            << "\nReal Value: " << value << std::endl
            << std::endl;
  std::cout << "Absolute: " << absoluteError(approx, value) << std::endl;
}
{% endhighlight %}

## Result

```
Approximate: 3.2
Real Value: 3.14159

Absolute: 0.05841
```

---
## Input relativeError

`relativeError<T>(T approx, T value)` requires:
* `T approx` - the approximated value
* `T value` the real value

 elative error \\(= \eta := \frac{\epsilon}{\lvert v \rvert} \\)

**Note:** `approx` and `value` must be the same type.

## Output

`relativeError` returns a `T` with the relative error.

## Code
{% highlight c++ %}
template <typename T>
inline T relativeError(const T approx, const T value)
{
  return std::abs(absoluteError(value, approx) / value);
}
{% endhighlight %}

## Example
{% highlight c++ %}
int main()
{
  double approx = 3.2;
  double value = 3.14159;

  std::cout << "Approximate: " << approx
            << "\nReal Value: " << value << std::endl
            << std::endl;
  std::cout << "Relative: " << relativeError(approx, value) << std::endl;
}
{% endhighlight %}

## Result
```
Approximate: 3.2
Real Value: 3.14159

Relative: 0.0185925
```

**Last Modification date:** 11 January 2018
