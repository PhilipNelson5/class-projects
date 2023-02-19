#pragma once

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#include <cmath>
#include <iostream>

namespace Math {

/**
 * @brief A 2D vector
 * 
 * @tparam T type of vector
 */
template <typename T>
struct Vec2
{
    T x, y;
};

/**
 * @brief ostream insertion operator with format V(x, y)
 * 
 * @tparam T 
 * @param o ostream
 * @param v vector
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<< (std::ostream& o, Vec2<T> const & v)
{
    o << "V(" << v.x << ", " << v.y << ")";
    return o;
}

/**
 * @brief multiply a scalar and a vector
 * 
 * @tparam T 
 * @param a scalar
 * @param v vector
 * @return Vec2<T> result vector
 */
template <typename T>
CUDA_HD Vec2<T> operator*(const T a, const Vec2<T>& v)
{
    return { a * v.x, a * v.y };
}

/**
 * @brief divide a vector by a scalar
 * 
 * @tparam T 
 * @param v vector
 * @param a scalar
 * @return Vec2<T> result vector
 */
template <typename T>
CUDA_HD Vec2<T> operator/(const Vec2<T>& v, const T a)
{
    return { v.x / a, v.y / a };
}

/**
 * @brief add two vectors elementwise
 * 
 * @tparam T 
 * @param v1 vector
 * @param v2 vector
 * @return Vec2<T> result vector
 */
template <typename T>
CUDA_HD Vec2<T> operator+(Vec2<T> const & v1, Vec2<T> const & v2)
{
    return { v1.x + v2.x, v1.y + v2.y };
}

/**
 * @brief subtract two vectors elementwise
 * 
 * @tparam T 
 * @param v1 vector
 * @param v2 vector
 * @return Vec2<T> result vector
 */
template <typename T>
CUDA_HD Vec2<T> operator-(Vec2<T> const & v1, Vec2<T> const & v2)
{
    return { v1.x - v2.x, v1.y - v2.y };
}

/**
 * @brief compute the magnitude of a vector
 * 
 * @tparam T 
 * @param v vector
 * @return double magnitude 
 */
template <typename T>
CUDA_HD double magnitude(Vec2<T> const & v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

/**
 * @brief normalize a vector
 * 
 * @tparam T 
 * @param v vector
 * @return Vec2<T> normalized vector
 */
template <typename T>
CUDA_HD Vec2<T> normalize(Vec2<T> const & v)
{
    const T m = magnitude(v);
    if (m > 0) return v / m;
    return v;
    // throw std::domain_error("attempt to normalize vector with magnitude zero");
}

/**
 * @brief rotate a vector by degrees
 * 
 * @tparam T 
 * @param v vector
 * @param deg degrees
 * @return Vec2<T> result vector
 */
template <typename T>
Vec2<T> rotate(Vec2<T> const & v, const T deg)
{
    const T rad = deg * M_PI / 180;
    return { 
        v.x * std::cos(rad) - v.y * std::sin(rad),
        v.x * std::sin(rad) + v.y * std::cos(rad)
    };
}
       
} // namespace Math
