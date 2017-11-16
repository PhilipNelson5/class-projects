#ifndef EXPECTED_HPP
#define EXPECTED_HPP

#include <exception>
#include <functional>
#include <iostream>

template <typename T>
class Expected
{
private:
  union State{
    T t;
    std::exception_ptr e;

  } data;

  bool good;

public:
  Expected(T t) : data({.t = {t}}), good(true) {}
  Expected(std::exception_ptr p) : data({.e = {p}}), good(false) {}
  Expected(std::exception e) : data({.e = {std::make_exception_ptr(e)}}), good(false) {}

  T value(void) const
  {
    if (good) return data.t;
    std::rethrow_exception(data.e);
  }

  operator T() { return value(); }

  template <typename U>
  Expected<U> apply(std::function<U(T)> f)
  {
    if (!good) return data.e;
    try
    {
      return f(data.t);
    }
    catch (...)
    {
      return std::current_exception();
    }
  }
};

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
