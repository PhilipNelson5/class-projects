#pragma once

#include <Vector/Vec2.hpp>
#include <Point/Pnt2.hpp>
#include <Expected/Expected.hpp>

namespace Math {

/**
 * @brief Euler method for calculating next point in function with a small step
 * 
 * @tparam T 
 * @tparam F 
 * @param f function
 * @param p point
 * @param h step
 * @return Expected<Pnt2<T>> next point
 */
template <typename T, typename F>
Expected<Pnt2<T>> euler(const F& f, Pnt2<T> p, const T h)
{
    return p + (h * f(p));
}

} // namespace Math
