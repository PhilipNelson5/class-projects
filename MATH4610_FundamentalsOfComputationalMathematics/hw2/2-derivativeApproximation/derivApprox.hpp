#ifndef DERIVATIVE_APPROXIMATION_HPP
#define DERIVATIVE_APPROXIMATION_HPP

/**
 * approximates the derivative of a function at x
 * df/dx = [ d(x+h) - f(x) ] / h
 *
 * @tparam T The type of x, h, and the function f's param and return
 * @param f  The T(T) function to approximate
 * @param x  The point to approximate the function at
 * @param h  The value of h to use in the approximation
 * @return   The approximation of the derivative f at x
 */
template <typename T, typename F>
inline T deriv_approx(F f, T x, T h)
{
  return (f(x + h) - f(x)) / h;
}

#endif
