#include "warmingAndBeam.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  constexpr double xDomain[] = {0.0, 1.0};
  constexpr double tDomain[] = {0.0, 1.0};
  // constexpr auto dt = 1.0e-3;
  // constexpr auto dx = 1.0e-3;
  constexpr auto dt = .1;
  constexpr auto dx = .1;
  constexpr std::size_t S = ((xDomain[1] - xDomain[0]) / dx);
  auto c = 0.7;
  auto eta = [](const double& x) -> double {
    return (x >= 0.3 && x <= 0.6) ? 100 : 0;
  };

  auto solution = warming_and_beam<S, double>(xDomain, tDomain, dx, dt, eta, c);

  for (auto i = 0u; i < solution.size(); ++i)
  {
    for (auto j = 0u; j < solution[i].size(); ++j)
    {
      std::cout << std::setprecision(3) << std::setw(10) << solution[j][i]
                << " ";
    }
    std::cout << '\n';
  }

  return EXIT_SUCCESS;
}
