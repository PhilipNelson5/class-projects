#pragma once
#include <algorithm>
#include <complex>
#include <function/function.hpp>
#include <utility>
#include <vector>

namespace aberth
{
std::vector<std::complex<double>> findRoots(const Function::Function &f);
} // namespace aberth
