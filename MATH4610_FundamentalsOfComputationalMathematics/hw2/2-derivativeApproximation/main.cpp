#include "../1-error/error.hpp"
#include "derivApprox.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

double poly(double x)
{
  return x*x*x + 2*x*x + 10*x + 7;
}

int main()
{
  auto x = 3.0;
  auto h = 1.0;
  auto value = 3*x*x + 4*x + 10;
  auto f = poly;
  auto approx = 0.0;
  auto error_abs = 0.0;
  auto error_rel = 0.0;
  auto approx_0 = 0.0;

  do
  {
    approx_0 = approx;
    approx = deriv_approx(f, x, h);
    error_abs = absolute_error(approx, value);
    error_rel = relative_error(approx, value);
    std::cout << h << ' ' << std::setprecision(10) << value << ' ' << approx
              << ' ' << error_abs << ' ' << error_rel << '\n';
    h /= 2;
  } while (approx_0 != approx);
}
