#include "../1-error/error.hpp"
#include "../2-derivativeApproximation/derivApprox.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

double fact(int i)
{
  auto f = 1;
  for (; i > 1; --i)
    f *= i;
  return f;
}

double taylorSin(const double x, const int kmax)
{
  double sum = 0;
  for(auto k = 0; k < kmax; ++k)
  {
    sum += std::pow(-1, k) * std::pow(x, 2*k+1) / fact(2*k+1);
  }

  return sum;
}

int main()
{
  const auto x = 0.0;
  auto h = 1.0;
  // const auto value = pow(x, -.5)/2;
  // const auto f = sqrt;
  // const auto newApprox = [](double x, double h){return 1/(sqrt(x+h) + sqrt(x));};
  auto approx = 0.0;
  const auto value = 1.0;
  auto error_abs = 0.0;
  auto error_rel = 0.0;
  // auto approx_0 = 0.0;
  auto h0 = 0.0;
  const auto tsin = [](double x){return taylorSin(x, 5);};

  do
  {
    // approx_0 = approx;
    h0 = h;

    // approx = deriv_approx(f, x, h);
    // approx = newApprox(x, h);
    approx = deriv_approx(tsin, x, h);
    error_abs = absolute_error(approx, value);
    error_rel = relative_error(approx, value);
    std::cout << h << ' ' << std::setprecision(10) << value << ' ' << approx
              << ' ' << error_abs << ' ' << error_rel << '\n';
    h /= 2;
  } while (h < h0);
}
