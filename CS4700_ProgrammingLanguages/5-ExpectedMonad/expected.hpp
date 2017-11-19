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

#define MixedMode(op)\
template <typename T, typename V>\
auto operator op (Expected<T> t, Expected<V> v)\
{\
  return t.template apply<decltype(std::declval<T>() op v)>(\
    [&](T myT) { return myT op v; });\
}\
template <typename T, typename V>\
auto operator op (Expected<T> t, V v)\
{\
  return t.template apply<decltype(std::declval<T>() op v)>(\
    [&](T myT) { return myT op v; });\
}\
template <typename T, typename V>\
auto operator op (V v, Expected<T> t)\
{\
  return t.template apply<decltype(v op std::declval<T>())>(\
    [&](T myT) { return v op myT; });\
}\

MixedMode(+)
MixedMode(-)
MixedMode(*)
MixedMode(/)
MixedMode(%)
MixedMode(>)
MixedMode(>=)
MixedMode(<)
MixedMode(<=)
MixedMode(==)

template <typename T>
std::ostream& operator<<(std::ostream& o, Expected<T> e)
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
