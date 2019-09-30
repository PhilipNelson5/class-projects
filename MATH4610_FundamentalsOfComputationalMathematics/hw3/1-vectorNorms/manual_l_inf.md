---
title: l_inf
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# l_inf Software Manual
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

**Routine Name:** l_inf

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./norms.out** that can be executed.

**Description/Purpose:** This is a template function that can be used to calculate the \\(l_\infty\\) norm of any a vector of any type s.t. \\(\|\| v \|\|_\infty = max \{\|x_1\|, \|x_2\|, \cdots , \|x_n\|\}\\)

**Input:** The function takes on argument, the vector

**Output:** The \\(l_\infty\\) norm of \\(v\\)

**Usage/Example:**

``` c++
int main()
{
  std::vector<double> v{3, 4, 1};
  std::cout << "v : " << v << '\n';
  std::cout << "l_inf norm : " <<  l_inf(v) << '\n';
}
```

**Output** from the lines above
```
v : [          3         4         1 ]

l_inf norm : 4
```

_explanation of output_:
The first line is the vector

The second line is the \\(l_\infty\\) norm of \\(v\\)

**Implementation/Code:** The following is the code for `l_inf`

The implementation of `l_inf` uses [std::max_element](https://en.cppreference.com/w/cpp/algorithm/max_element) which takes a lambda function that compares two elements by their absolute values using [std::abs](https://en.cppreference.com/w/cpp/numeric/math/abs). Once the absolute largest element has been identified, it's absolute value is returned. This is the definition of the infinity norm.

``` c++
/**
 * Determine the l_pNorm of a vector
 *
 * @tparam T The type of the elements in `a`
 * @param a  The vector
 * @param p  The `p` of the l_pNorm
 */
template <typename T>
inline T l_inf(std::vector<T> const& a)
{
  return std::abs(
      *std::max_element(
        begin(a), end(a), [](T const a, T const b) {
          return std::abs(a) < std::abs(b);
        })
      );
}
```

**Last Modified:** September 2018
