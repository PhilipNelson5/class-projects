#include "../logistic/logistic.hpp"
#include "../explicitEuler/explicitEuler.hpp"
#include "../5.1IVP/firstOrderIVP.hpp"
#include <iostream>
#include <vector>

int main() {
  double dt = 0.00001;

  double alpha = 10.0;
  std::vector lambdas = { 1.0, -1.0, 100.0 };

  double gamma = 0.1;
  double beta = 0.0001;
  std::vector Pos = { 25.0, 40000.0 };

  std::cout << "----- Lambda Differential Equation -----" << std::endl;
  for (const auto lambda : lambdas) {
    std::cout << explicit_euler( alpha, beta, gamma, dt,
        [=](double a, double b) {
        (void)b;
          return lambda * a;
        }
    ) << '\n';

    auto solveIVP = firstOrderIVPSolver(lambda, alpha);

    std::cout << solveIVP(dt);
  }

  std::cout << std::endl;
  std::cout << "----- Logistic Differential Equation -----" << std::endl;
  for (const auto p0 : Pos) {
    std::cout << explicit_euler( alpha, beta, gamma, dt,
        [=](double a, double b) {
        (void)b;
        return gamma * a - beta * a * a;
        }
    ) << '\n';

    std::cout << logistic(alpha, beta, dt, p0) << '\n';
  }

  return EXIT_SUCCESS;
}
