---
title: Inverse Power Iteration Example on the Elliptic ODE
math: true
layout: default
---

{% include mathjax.html %}

<a href="https://philipnelson5.github.io/MATH5620/SoftwareManual"> Table of Contents </a>
# Example of Inverse Power Iteration

**Author:** Philip Nelson

**Language:** C++

## Description

This demonstrates the condition number method on the tri-diagonal matrix used to solve the Elliptic ODE.

## Output

The condition number of the `N`x`N` matrix with increasing `N`. Observe that as the size of the matrix increases, the condition number increases unbounded

## Example
{% highlight c++ %}
int main()
{
  std::cout << "A\n" << SecOrdFinDifMethEllipticMat<double, 5>() << std::endl << std::endl;
  std::cout << "3x3" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 3>()) << std::endl << std::endl;
  std::cout << "5x5" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 5>()) << std::endl << std::endl;
  std::cout << "10x10" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 10>()) << std::endl << std::endl;
  std::cout << "15x15" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 15>()) << std::endl << std::endl;
  std::cout << "25x25" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 25>()) << std::endl << std::endl;
  std::cout << "75x75" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 75>()) << std::endl << std::endl;
  std::cout << "100x100" << std::endl;
  std::cout << conditionNumber(SecOrdFinDifMethEllipticMat<double, 100>()) << std::endl << std::endl;
}
{% endhighlight %}

## Result
```
A
| -2     1     0  |
|  1    -2     1  |
|  0     1    -2  |

3x3
-5.82843

5x5
-13.9282

10x10
-48.3742

15x15
-103.087

25x25
-273.306

75x75
-2340.09

100x100
-4133.2

```

**Last Modification date:** 27 February 2018
