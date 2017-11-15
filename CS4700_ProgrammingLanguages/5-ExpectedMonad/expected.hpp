#ifndef EXPECTED_HPP
#define EXPECTED_HPP

#include <exception>
#include <iostream>

template <typename T>
class Expected
{
  T data;
  std::exception_ptr error;

public:
  Expected(T t) : data(t), error(nullptr) {}

  T value(void)
  {
    if (error)
      throw error;
    else
      return data;
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& o, Expected<T> & e)
{
  try
  {
    o << e.value();
  }
  catch(std::exception const & e)
  {
    o << e.what();
  }

  return o;
}

#endif
