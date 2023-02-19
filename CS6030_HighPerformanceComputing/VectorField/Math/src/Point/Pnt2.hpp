#pragma once

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#include <iostream>
#include <cmath>

#include <Vector/Vec2.hpp>

namespace Math {

/**
 * @brief A 2D point
 * 
 * @tparam T type of point
 */
template <typename T>
struct Pnt2
{
    T x, y;
};

/**
 * @brief ostream insertion operator with format P(x, y)
 * 
 * @tparam T 
 * @param o ostream
 * @param p point
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<< (std::ostream& o, Pnt2<T> const & p)
{
    o << "P(" << p.x << ", " << p.y << ")";
    return o;
}

/**
 * @brief add two points component wise
 * 
 * @tparam T 
 * @param p1 point 1
 * @param p2 point 2 
 * @return Pnt2<T> result point 
 */
template <typename T>
CUDA_HD Pnt2<T> operator+(Pnt2<T> const & a, Vec2<T> const & b)
{
    return { a.x + b.x, a.y + b.y };
}

/**
 * @brief calculate cartesian distance between two points
 * 
 * @tparam T 
 * @param a point 1
 * @param b point 2
 * @return T cartesian distance
 */
template <typename T>
T distance(Pnt2<T> const & a, Pnt2<T> const & b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
    
} // namespace Math

namespace std {

/**
 * @brief ceiling of x and y component of point
 * 
 * @tparam T 
 * @param p point
 * @return Math::Pnt2<T> result point
 */
template <typename T>
Math::Pnt2<T> ceil(const Math::Pnt2<T> p)
{
    return { std::ceil(p.x), std::ceil(p.y) };
}

/**
 * @brief floor of x and y component of point
 * 
 * @tparam T 
 * @param p point
 * @return Math::Pnt2<T> result point
 */
template <typename T>
Math::Pnt2<T> floor(const Math::Pnt2<T> p)
{
    return { std::floor(p.x), std::floor(p.y) };
}

} // namespace std
