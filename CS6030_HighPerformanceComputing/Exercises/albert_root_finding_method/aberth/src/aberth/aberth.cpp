#include "aberth.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>

namespace
{

double random_double(double low, double high)
{
    static std::random_device rd;
    static std::mt19937 mt(rd());
    std::uniform_real_distribution<> dist(low, high);
    return dist(mt);
}
} // namespace

namespace aberth
{

std::vector<std::complex<double>> initRoots(const Function::Function &f)
{
    const int degree = Function::degree(f);
    const auto [upper, lower] = Function::getUpperAndLowerBounds(f);
    std::cout << "upper/lower/degree: " << upper << " " << lower << " " << degree << std::endl;

    std::vector<std::complex<double>> roots;
    for (int i = 0; i < degree; ++i)
    {
        const double radius = std::sqrt(random_double(lower, upper));
        const double theta = random_double(0, 2 * M_PI);
        roots.emplace_back(radius * std::cos(theta), radius * std::sin(theta));
    }
    return roots;
}

std::vector<std::complex<double>> findRoots(const Function::Function &f)
{
    auto roots = initRoots(f);
    auto valid = 0u;
    do
    {
        for (auto i = 0u; i < roots.size(); ++i)
        {
            auto &r = roots[i];
            // std::cout << "root: " << r << std::endl;
            const auto ratio = Function::image(f, r) / Function::derivative(f, r);
            // std::cout << "image/derivative: " << Function::image(f, r) << " " << Function::derivative(f, r)
            //   << std::endl;
            const auto sum = std::accumulate(
                begin(roots), end(roots), std::complex<double>(0.0),
                [&r](std::complex<double> acc, std::complex<double> z) { return z != r ? acc + 1.0 / (r - z) : acc; });
            const auto offset = ratio / (1.0 - (ratio * sum));
            // std::cout << "ratio/sum/offset: " << ratio << " " << sum << " " << offset << std::endl;
            if (offset.real() < std::numeric_limits<double>::epsilon() &&
                offset.imag() < std::numeric_limits<double>::epsilon())
            {
                ++valid;
            }
            r -= offset;
        }
        // break;
    } while (valid != roots.size());
    return roots;
}
} // namespace aberth
