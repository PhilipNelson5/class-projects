#ifndef RETURN_STATMENT_NODE_HPP
#define RETURN_STATMENT_NODE_HPP

#include "ExpressionNode.hpp" // for ExpressionNode
#include "StatementNode.hpp"  // for StatementNode

#include <memory> // for shared_ptr
#include <string> // for string

class ReturnStatementNode : public StatementNode
{
public:
  ReturnStatementNode()
    : expr(nullptr)
  {}

  ReturnStatementNode(ExpressionNode*& expr)
    : expr(expr)
  {}

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

  const std::shared_ptr<ExpressionNode> expr;
};

#endif
