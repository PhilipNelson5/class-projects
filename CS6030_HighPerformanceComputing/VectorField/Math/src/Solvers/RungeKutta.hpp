#pragma once

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#include <Vector/Vec2.hpp>
#include <Point/Pnt2.hpp>
#include <Expected/Expected.hpp>

#include <functional>
namespace {

using Math::Vec2;
using Math::Pnt2;

/**
 * @brief First Runge Kutta term
 * 
 * @tparam T 
 * @tparam F 
 * @param f function
 * @param p point
 * @return Expected<Vec2<T>> first term
 */
template <typename T, typename F>
CUDA_HD Expected<Vec2<T>> k1(const F& f, Pnt2<T> p)
{
    return f(p) >>= Math::normalize<T>;
}

/**
 * @brief Second Runge Kutta term
 * 
 * @tparam T 
 * @tparam F 
 * @param f function
 * @param p point
 * @param h step
 * @param k_1 first Runge Kutta term
 * @return Expected<Vec2<T>> second term
 */
template <typename T, typename F>
CUDA_HD Expected<Vec2<T>> k2(const F& f, Pnt2<T> p, const T h, const Vec2<T> k_1)
{
    const T x1 = p.x + h / 2.0 * k_1.x;
    const T y1 = p.y + h / 2.0 * k_1.y;
    return f(Pnt2<T>{ x1, y1 }) >>= Math::normalize<T>;
}

/**
 * @brief Third Runge Kutta term
 * 
 * @tparam T 
 * @tparam F 
 * @param f function
 * @param p point
 * @param h step
 * @param k_2 second Runge Kutta term
 * @return Expected<Vec2<T>> third term
 */
template <typename T, typename F>
CUDA_HD Expected<Vec2<T>> k3(const F& f, Pnt2<T> p, const T h, const Vec2<T> k_2)
{
    const T x1 = p.x + h / 2.0 * k_2.x;
    const T y1 = p.y + h / 2.0 * k_2.y;
    return f(Pnt2<T>{ x1, y1 }) >>= Math::normalize<T>;
}

/**
 * @brief Fourth Runge Kutta term
 * 
 * @tparam T 
 * @tparam F 
 * @param f function
 * @param p point
 * @param h step
 * @param k_3 third Runge Kutta term
 * @return Expected<Vec2<T>> fourth term
 */
template <typename T, typename F>
CUDA_HD Expected<Vec2<T>> k4(const F& f, Pnt2<T> p, const T h, const Vec2<T> k_3)
{
    const T x1 = p.x + h * k_3.x;
    const T y1 = p.y + h * k_3.y;
    return f(Pnt2<T>{ x1, y1 }) >>= Math::normalize<T>;
}
    
} // namespace


namespace Math
{

/**
 * @brief Runge Kutta method for calculating next point in function with a small step
 * 
 * @tparam T 
 * @tparam F 
 * @param f function 
 * @param p point
 * @param h step
 * @return Expected<Pnt2<T>> next point
 */
template <typename T, typename F>
CUDA_HD Expected<Pnt2<T>> runge_kutta(const F& f, Pnt2<T> p, const T h)
{
    const Expected<Vec2<T>> k_1 = k1(f, p);
    const Expected<Vec2<T>> k_2 = k_1 >>= [&](Vec2<T> t){ return k2(f, p, h, t); };
    const Expected<Vec2<T>> k_3 = k_2 >>= [&](Vec2<T> t){ return k3(f, p, h, t); };
    const Expected<Vec2<T>> k_4 = k_3 >>= [&](Vec2<T> t){ return k4(f, p, h, t); };
    return p + h / T(6.0) * (k_1 + T(2.0) * k_2 + T(2.0) * k_3 + k_4);
}

} // namespace Math
