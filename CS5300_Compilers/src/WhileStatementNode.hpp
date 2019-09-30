#ifndef WHILE_STATEMENT_NODE_HPP
#define WHILE_STATEMENT_NODE_HPP

#include "ExpressionNode.hpp"
#include "ListNode.hpp"
#include "StatementNode.hpp"

#include <memory>

class WhileStatementNode : public StatementNode
{
public:
  WhileStatementNode(ExpressionNode*& expr, ListNode<StatementNode>*& statements);

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

  const std::shared_ptr<ExpressionNode> expr;
  const std::vector<std::shared_ptr<StatementNode>> statements;
};

#endif
