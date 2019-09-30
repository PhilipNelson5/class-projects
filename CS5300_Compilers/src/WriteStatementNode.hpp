#ifndef WRITE_STATEMENT_NODE_HPP
#define WRITE_STATEMENT_NODE_HPP

#include "ExpressionNode.hpp"
#include "ListNode.hpp"
#include "StatementNode.hpp"

#include <memory>
#include <vector>

class WriteStatementNode : public StatementNode
{
public:
  WriteStatementNode(ListNode<ExpressionNode>*& exprList);
  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

private:
  const std::vector<std::shared_ptr<ExpressionNode>> expressionList;
};

#endif
