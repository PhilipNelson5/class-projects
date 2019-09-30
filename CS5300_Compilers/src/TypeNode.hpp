#ifndef TYPE_NODE_HPP
#define TYPE_NODE_HPP

#include <memory> // for shared_ptr, operator==
#include <string> // for string

class Type;

class TypeNode
{
public:
  TypeNode(std::string id)
    : id(id)
    , type(nullptr)
  {}

  TypeNode(std::shared_ptr<Type> type)
    : id("")
    , type(type)
  {}

  const std::shared_ptr<Type> getType();

  const std::string id;

private:
  std::shared_ptr<Type> type;
};
#endif
