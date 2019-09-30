#include "ExpressionNode.hpp"

ExpressionNode::ExpressionNode()
  : type(nullptr)
{}

ExpressionNode::ExpressionNode(std::shared_ptr<Type> type)
  : type(type)
{}

const std::shared_ptr<Type> ExpressionNode::getType()
{
  return type;
}

void ExpressionNode::setType(std::shared_ptr<Type> newType)
{
  type = newType;
}

bool ExpressionNode::isLiteral() const
{
  return false;
}
