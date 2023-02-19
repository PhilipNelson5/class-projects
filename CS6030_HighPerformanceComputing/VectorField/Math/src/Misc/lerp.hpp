#pragma once

namespace Math {

/**
 * @brief linear interpolation
 * 
 * @tparam T 
 * @param a start range
 * @param b end range
 * @param t percentage along range
 * @return T linearly interpolated value
 */
template <typename T>
inline T lerp(const T a, const T b, const T t)
{
    return a + t * (b - a);
}

} // namespace Math
