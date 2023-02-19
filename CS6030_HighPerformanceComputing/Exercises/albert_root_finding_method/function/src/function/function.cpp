#include "function.hpp"

namespace Function
{

Function::Function(std::vector<std::pair<int, double>> ilist)
{
    std::sort(begin(ilist), end(ilist),
              [](const std::pair<int, std::complex<double>> &a, const std::pair<int, std::complex<double>> &b) -> int {
                  return a.first < b.first;
              });

    for (const auto &elem : ilist)
        coef.emplace_back(elem.first, elem.second);
}

int degree(const Function &f)
{
    return f.coef.back().first;
}

std::complex<double> image(const Function &f, const std::complex<double> x)
{
    std::complex<double> val = 0.0;
    for (const auto &[i, coef] : f.coef)
    {
        val += coef * std::pow(x, i);
    }
    return val;
}

std::complex<double> derivative(const Function &f, const std::complex<double> x)
{
    return (image(f, x + 1e-12) - image(f, x - 1e-12)) / 2e-12;
}

std::tuple<double, double> getUpperAndLowerBounds(const Function &f)
{
    const auto &coef = f.coef;
    const double upper = 1.0 + 1.0 / std::abs(coef.back().second) *
                                   std::abs(std::max_element(begin(coef), end(coef) - 1,
                                                             [](const std::pair<int, std::complex<double>> &a,
                                                                const std::pair<int, std::complex<double>> &b) {
                                                                 return std::abs(a.second) > std::abs(b.second);
                                                             })
                                                ->second);
    const double lower = 1.0 / upper;
    // const double lower =
    //     std::abs(coef[0].second) /
    //     (std::abs(coef[0].second) + std::abs(std::max_element(begin(coef) + 1, end(coef),
    //                                                           [](const std::pair<int, std::complex<double>> &a,
    //                                                              const std::pair<int, std::complex<double>> &b) {
    //                                                               return std::abs(a.second) > std::abs(b.second);
    //                                                           })
    //                                              ->second));
    return std::make_tuple(upper, lower);
}

} // namespace Function