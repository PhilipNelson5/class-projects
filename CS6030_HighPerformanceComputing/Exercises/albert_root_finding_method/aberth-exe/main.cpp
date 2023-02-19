#include "aberth/aberth.hpp"
#include "function/function.hpp"
#include <iostream>

int main()
{
    Function::Function f({{0, 5.0}, {1, -3.0}, {2, -2.0}, {3, 1.0}});
    std::cout << "f(3) = " << Function::image(f, 3) << std::endl;
    std::cout << "f(4) = " << Function::image(f, 4) << std::endl;
    std::cout << "f`(-0.53518) = " << Function::derivative(f, 0.53518) << std::endl;
    std::cout << "f`(1.8685) = " << Function::derivative(f, 1.8685) << std::endl;
    const auto roots = aberth::findRoots(f);
    for (const auto &root : roots)
        std::cout << root << std::endl;
}