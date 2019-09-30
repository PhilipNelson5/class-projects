#ifndef SUBSCRIPT_OPERATOR_NODE_HPP
#define SUBSCRIPT_OPERATOR_NODE_HPP

#include "LvalueNode.hpp" // for LvalueNode
#include "Value.hpp"      // for Value

#include <memory> // for shared_ptr
#include <string> // for string
class ExpressionNode;

class SubscriptOperatorNode : public LvalueNode
{
public:
  SubscriptOperatorNode(LvalueNode* lValue, ExpressionNode* expr);
  virtual bool isConstant() const override;
  virtual const std::shared_ptr<Type> getType() override;
  virtual std::string getId() const override;
  virtual std::variant<std::monostate, int, char, bool> eval() const override;
  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;

  const std::shared_ptr<LvalueNode> lValue;
  const std::shared_ptr<ExpressionNode> expr;
};

#endif
