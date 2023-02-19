#pragma once

#include <vector>

/**
 * @brief 2D vector stored in contiguous memory
 * 
 * @tparam T type of the elements
 */
template <typename T>
class Vector2D
{
public:
  Vector2D(const unsigned n_cols, std::initializer_list<T> l): n_cols(n_cols), data(l) {}

  T* operator[](const unsigned i)
  {
    return &this->data[i*n_cols];
  }

  unsigned size() { return this->data.size(); }
  unsigned rows() { return this->data.size() / this->n_cols; }
  unsigned cols() { return this->n_cols; }

  unsigned n_cols;
  std::vector<T> data;
};

template<typename T>
std::ostream& operator<<(std::ostream& o, Vector2D<T> v)
{
  o << "[\n";
  for (unsigned i = 0; i < v.rows(); ++i)
  {
    o << "  ";
    for (unsigned j = 0; j < v.cols(); ++j)
      o << v[i][j] << ' ';
    o << '\n';
  }
  o << "]";
  return o;
}
