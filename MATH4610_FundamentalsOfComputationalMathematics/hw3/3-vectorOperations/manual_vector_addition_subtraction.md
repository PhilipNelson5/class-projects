---
title: Vector Addition and Subtraction
layout: default
math: true
---
{% include mathjax.html %}
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Vector Addition Software Manual

**Routine Name:** Vector Addition and Subtraction

**Author:** Philip Nelson

**Language:** C++. The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./vectorOps.out** that can be executed.

**Description/Purpose:** This routine overloads the `+` and `-` operators in c++ allowing two vectors to be added and subtracted with the following syntax, `a + b` and `a - b`.

**Input:** The operator requires two operands, `a` and `b`, where `a, b` are `std::vector<T>`

```
@tparam T Type of the elements in the first vector
@tparam U Type of the elements in the second vector
@tparam R Type of the elements in the result vector
@param a  The first vector
@param b  The second vector
```

**Output:** A vector with the result of vector addition or subtraction with the two vector operands.

**Usage/Example:**

``` cpp
int main()
{
  std::vector<double> a = {-1.1, 2.3, 3.5};
  std::vector<double> b = {4.2, 5.4, 6.6};
  std::cout << "a\t" << a << '\n';
  std::cout << "b\t" << b << '\n';
  std::cout << "a + b\t" << a + b << '\n';
  std::cout << "a - b\t" << a - b << '\n';
}
```

**Output** from the lines above
```
a      [        1.1       2.3       3.5 ]

b      [        4.2       5.4       6.6 ]

a + b  [        5.3       7.7      10.1 ]

a - b  [       -3.1      -3.1      -3.1 ]
```

_explanation of output_:

The first two lines display two vectors `a` and `b`.

The third line is the result of `a + b` and the fourth line is the result of `a - b`.

**Implementation/Code:** The following is the code for vector addition and subtraction

For code re-usability, the implementation takes advantage of the c++ preprocessor to generate the actual addition and subtraction code. The difference between adding and subtracting vectors is simply the difference between using the plus operator or the minus operator. Hence, `vector_add_subtract` defines the generic form of the operation with the operator replaced with a variable. This way `vector_add_subtract(+) vector_add_subtract(-)` can fill in the operator and the preprocessor will generate the code, no branching structure or duplicate code required.

``` cpp
#define vector_add_subtract(op)                                                \
  template <typename T, typename U, typename R = decltype(T() op U())>         \
  std::vector<R> operator op(std::vector<T> const& a, std::vector<U> const& b) \
  {                                                                            \
    // check the sizes are equal                                               \
    if (a.size() != b.size())                                                  \
    {                                                                          \
      std::cerr << "ERROR: bad size in vector addition\n";                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
                                                                               \
    // initalize it result vector                                              \
    std::vector<R> result(a.size());                                           \
                                                                               \
    // add or subtract the vectors elementwise                                 \
    for (auto i = 0u; i < a.size(); ++i)                                       \
    {                                                                          \
      result[i] = a[i] op b[i];                                                \
    }                                                                          \
                                                                               \
    // return the result                                                       \
    return result;                                                             \
  }

// Call the macro with addition and subtraction operator
vector_add_subtract(+) vector_add_subtract(-)
```

**Last Modified:** October 2018
