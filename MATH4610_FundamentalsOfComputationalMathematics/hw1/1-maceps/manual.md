---
title: Machine Epsilon
layout: default
---
<a href="https://philipnelson5.github.io/math4610/SoftwareManual"> Table of Contents </a>
# Maceps Software Manual

**Routine Name:** `maceps`

**Author:** Philip Nelson

**Language:** C++.

The code can be compiled using the GNU C++ compiler (gcc). A make file is included to compile an example program

For example,

```
make
```

will produce an executable **./maceps.out** that can be executed.

**Description/Purpose:** This routine will compute the precision value for the machine epsilon or the number of digits
in the representation of real numbers for any numeric type. This is a routine for analyzing the behavior of any computer. This
usually will need to be run one time for each computer.

**Input:** The function requires one template parameter, the numeric type you are testing ex. int, long, float, double, long double.

**Output:** This routine returns a tuple containing the precision value for the number of decimal digits that can be represented on the computer being queried and the machine epsilon of the type.

**Usage/Example:**
The routine is simply called with the template argument and no parameters. Since the routine returns a `std::tuple`, we can use structured bindings to assign the precision and machine epsilon to different variables after the function call.

``` c++
int main()
{
  auto [float_prec, float_eps] = maceps<float>();
  std::cout << "float\n";
  std::cout << "precision:\t" << float_prec << '\n';
  std::cout << "maceps:\t\t" << float_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<float>::epsilon()
            << "\n\n";

  auto [double_prec, double_eps] = maceps<double>();
  std::cout << "double\n";
  std::cout << "precision:\t" << double_prec << '\n';
  std::cout << "maceps:\t\t" << double_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<double>::epsilon()
            << "\n\n";

  auto [long_double_prec, long_double_eps] = maceps<long double>();
  std::cout << "long double\n";
  std::cout << "precision:\t" << long_double_prec << '\n';
  std::cout << "maceps:\t\t" << long_double_eps << '\n';
  std::cout << "std::numeric:\t" << std::numeric_limits<long double>::epsilon()
            << "\n\n";

  return EXIT_SUCCESS;
}
```

**Output** from the lines above

```
float
precision:	24
maceps:		1.19209e-07
std::numeric:	1.19209e-07

double
precision:	53
maceps:		2.22045e-16
std::numeric:	2.22045e-16

long double
precision:	64
maceps:		1.0842e-19
std::numeric:	1.0842e-19
```

The values labeled precision represent the number of binary digits that define the machine epsilon.
The values labeled maceps are related to the decimal version of the same value.
The values labeled std::numeric are the result of std::numeric\_limits<type>::epsilon, for comparison.

**Implementation/Code:** The following is the code for `maceps`

The code for maceps takes advantage of C++ templates to be able to write single, double and more with a single template function. This is advantageous because the implementation of maceps is essentially the same, only the type changes. Using templates allows the necessary types to be inserted and **multiple** functions are generated at _compile_ time. No branching occurs in the code.

``` c++
template <typename T>
std::tuple<int, T> maceps()
{
  T e = 1;
  T one = 1;
  T half = 0.5;
  int prec = 1;
  while (one + e * half > one)
  {
    e *= half;
    ++prec;
  }

  return std::make_tuple(prec, e);
}
```

If the following is written,

``` c++
int main()
{
  auto [float_prec, float_eps] = maceps<float>();

  auto [double_prec, double_eps] = maceps<double>();

  auto [long_double_prec, long_double_eps] = maceps<long double>();

  return EXIT_SUCCESS;
}
```

then the following is what the template instantiation would like from the compiler.

``` c++
template<> std::tuple<int, float> maceps<float>() {
    float e = 1;
    float one = 1;
    float half = 0.5;
    int prec = 1;
    while (one + e * half > one)
        {
            e *= half;
            ++prec;
        }
    return std::make_tuple(prec, e);
}
template<> std::tuple<int, double> maceps<double>() {
    double e = 1;
    double one = 1;
    double half = 0.5;
    int prec = 1;
    while (one + e * half > one)
        {
            e *= half;
            ++prec;
        }
    return std::make_tuple(prec, e);
}
template<> std::tuple<int, long double> maceps<long double>() {
    long double e = 1;
    long double one = 1;
    long double half = 0.5;
    int prec = 1;
    while (one + e * half > one)
        {
            e *= half;
            ++prec;
        }
    return std::make_tuple(prec, e);
}
```

As you can see, three **seperate** function are produced, one for each specified type.

**Last Modified:** September 2018
