# Machine Epsilon

**Routine Name:** maceps

**Author:** Philip Nelson

**Language:** C++

## Description

maceps returns the machine epsilon and precision of any primitive type. A make file is included with a driver program.

```
$ make
$ ./maceps.out
```

This will compile and run the driver program.

## Input

`maceps<T>( )` requires a template argument _T_ with the type of machine epsilon you want _( float, double, long double, etc... )_. Otherwise, `maceps` takes no input.

## Output

maceps returns an `eps` struct with members `int prec` which holds the precision and `T maceps` which holds the machine epsilon for the specified type.

## Code
``` C++
template <typename T>
eps<T> maceps()
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

  return eps(prec, e);
}
```

## Example
``` C++
int main()
{
  auto doubleeps = maceps<double>();
  std::cout << "double\n";
  std::cout << "precision:\t"    << doubleeps.prec << std::endl;
  std::cout << "maceps:\t\t"     << doubleeps.maceps << std::endl;
  std::cout << "std::numeric:\t" << std::numeric_limits<double>::epsilon() << std::endl << std::endl;
}
```

## Result
```
double
precision:	53
maceps:		2.22045e-16
std::numeric:	2.22045e-16
```

**Last Modification date:** 11 January 2018
