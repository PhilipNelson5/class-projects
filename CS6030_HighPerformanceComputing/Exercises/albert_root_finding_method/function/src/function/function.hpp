#pragma once
#include <algorithm>
#include <complex>
#include <tuple>
#include <vector>

namespace Function
{

struct Function
{
    Function(std::vector<std::pair<int, double>> ilist);
    /**
     * @brief A list of pairs representing the coefficients. The pairs are <degree, coefficient>.
     *
     */
    std::vector<std::pair<int, std::complex<double>>> coef;
};

int degree(const Function &f);
std::complex<double> image(const Function &f, const std::complex<double> x);
std::complex<double> derivative(const Function &f, const std::complex<double> x);
std::tuple<double, double> getUpperAndLowerBounds(const Function &f);

} // namespace Function