---
title: Power Iteration Example on the Elliptic ODE
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Example of Power Iteration

**Author:** Philip Nelson

**Language:** C++

## Description

This demonstrates the power iteration method on the tri-diagonal matrix used to solve the Elliptic ODE.

## Output

The largest Eigenvalue of the `N`x`N` matrix with increasing `N`. Observe that as the size of the matrix increases, the largest Eigenvalue asymptotically approaches the value 4.

## Example
{% highlight c++ %}
int main()
{
  std::cout << "A\n" << SecOrdFinDifMethEllipticMat<double, 5>() << std::endl << std::endl;
  std::cout << "5x5" << std::endl;
  std::cout << powerIteration(SecOrdFinDifMethEllipticMat<double, 5>(), 1000u) << std::endl << std::endl;
  std::cout << "10x10" << std::endl;
  std::cout << powerIteration(SecOrdFinDifMethEllipticMat<double, 10>(), 1000u) << std::endl << std::endl;
  std::cout << "100x100" << std::endl;
  std::cout << powerIteration(SecOrdFinDifMethEllipticMat<double, 100>(), 1000u) << std::endl << std::endl;
  std::cout << "1000x10000" << std::endl;
  std::cout << powerIteration(SecOrdFinDifMethEllipticMat<double, 1000>(), 1000u) << std::endl;
}
{% endhighlight %}

## Result
```
A
| -2     1     0  |
|  1    -2     1  |
|  0     1    -2  |

5x5
3.73205

10x10
3.91899

100x100
3.99852

1000x10000
3.99893
```

**Last Modification date:** 27 February 2018
