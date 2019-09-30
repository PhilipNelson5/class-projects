#ifndef SUBTRACT_NODE_HPP
#define SUBTRACT_NODE_HPP

#include "ExpressionNode.hpp" // for ExpressionNode
#include "Value.hpp"          // for Value

#include <memory> // for shared_ptr
#include <string> // for string

class SubtractNode : public ExpressionNode
{
public:
  SubtractNode(ExpressionNode*& left, ExpressionNode*& right);
  virtual bool isConstant() const override;
  virtual std::variant<std::monostate, int, char, bool> eval() const override;
  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;

private:
  const std::shared_ptr<ExpressionNode> lhs;
  const std::shared_ptr<ExpressionNode> rhs;
};

#endif
