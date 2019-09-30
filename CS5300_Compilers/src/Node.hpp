#ifndef NODE_HPP
#define NODE_HPP

#include "Value.hpp" // for Value

#include <string> // for string

class Node
{
public:
  virtual void emitSource(std::string indent) = 0;
  virtual ~Node() = default;
};

#endif
