#ifndef EXPECTED_HPP
#define EXPECTED_HPP

#include <exception>

template <typename T>
class Expected
{
  T data;
  std::exception_ptr error;

public:
  Expected(T t) : data(t), error(nullptr) {}

  T value(void)
  {
    if (!error)
      throw error;
    else
      return data;
  }
};

#endif
