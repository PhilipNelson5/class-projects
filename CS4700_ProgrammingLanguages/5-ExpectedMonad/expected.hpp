#ifndef EXPECTED_HPP
#define EXPECTED_HPP

#include <exception>
#include <functional>
#include <iostream>
#include <variant>

template <typename T>
class Expected
{
private:
  std::variant<T, std::exception_ptr> data;

public:
  Expected(T t) : data(t) {}
  Expected(std::exception_ptr p) : data(p) {}
  Expected(std::exception e) : data(std::make_exception_ptr(e)) {}

  T value(void) const
  {
    if (std::holds_alternative<T>(data)) return std::get<T>(data);
    std::rethrow_exception(std::get<std::exception_ptr>(data));
  }

  operator T() { return value(); }

  template <typename U>
  Expected<U> apply(std::function<U(T)> f)
  {
    if (std::holds_alternative<std::exception_ptr>(data))
      return std::get<std::exception_ptr>(data);
    try
    {
      return f(std::get<T>(data));
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
