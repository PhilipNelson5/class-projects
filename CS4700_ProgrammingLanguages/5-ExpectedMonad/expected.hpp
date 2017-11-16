#ifndef EXPECTED_HPP
#define EXPECTED_HPP

#include <exception>
#include <functional>
#include <iostream>

template <typename T>
class Expected
{
private:
  T t;
  std::exception_ptr e;
  bool good;

public:
  Expected(T t) : t(t), e(nullptr), good(true) {}
  Expected(std::exception_ptr p) : t(), e(p), good(false) {}
  Expected(std::exception e) : t(t), e(std::make_exception_ptr(e)), good(false) {}

  T value(void) const
  {
    if (good) return t;
    std::rethrow_exception(e);
  }

  operator T() { return value(); }

  Expected<T> apply(std::function<T(T)> f)
  {
    if (!good) return e;
    try
    {
      return f(t);
    }
    catch (...)
    {
      return std::current_exception();
    }
  }
};

template <typename T>
Expected<T> operator/(Expected<T> t, Expected<T> u)
{
  if (u == 0) return std::out_of_range("DIVIDE BY ZERO");
  return t.apply([u](T t) { return t / u; });
}

template <typename T>
Expected<T> operator+(Expected<T> t, Expected<T> u)
{
  return t.apply([u](T t) { return t + u; });
}

/*
// clang-format off
#define MixedMode(op)\
template <typename T, typename U>\
auto op(Expected<T> t, U u)\
{\
  return t.apply([u](T t) { return op(t,u); });\
}\
\
template <typename T, typename U>\
auto op(T t, Expected<U> u)\
{\
  return t.apply([t](U u) { return op(t,u); });\
}\
template <typename T, typename U>\
auto op(Expected<T> t, Expected<U> u)\
{\
  return t.apply([t](U u) { return op(t,u); });\
}\
// clang-format on

MixedMode(operator+)
MixedMode(operator-)
MixedMode(operator*)
MixedMode(operator/)
MixedMode(operator%)
MixedMode(operator>)
MixedMode(operator>=)
MixedMode(operator<)
MixedMode(operator<=)
MixedMode(operator==)
*/

template <typename T>
std::ostream& operator<<(std::ostream& o, Expected<T>& e)
{
  try
  {
    o << e.value();
  }
  catch (std::exception const& e)
  {
    o << e.what();
  }
  catch (...)
  {
    o << "UNKNOWN ERROR";
  }

  return o;
}

#endif
