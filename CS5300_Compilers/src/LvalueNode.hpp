#ifndef LVALUE_NODE_HPP
#define LVALUE_NODE_HPP

#include "ExpressionNode.hpp" // for ExpressionNode

#include <memory> // for shared_ptr
#include <string> // for string
class Type;

class LvalueNode : public ExpressionNode
{
public:
  LvalueNode()
    : ExpressionNode()
  {}

  LvalueNode(std::shared_ptr<Type> type)
    : ExpressionNode(type)
  {}

  virtual std::variant<std::monostate, int, char, bool> eval() const = 0;
  virtual bool isConstant() const = 0;
  virtual std::string getId() const = 0;
};

#endif
