#ifndef ASSIGNEMNT_STATEMENT_NODE_HPP
#define ASSIGNEMNT_STATEMENT_NODE_HPP

#include "LvalueNode.hpp"
#include "StatementNode.hpp" // for StatementNode
#include "Value.hpp"         // for Value

#include <memory> // for shared_ptr
#include <string> // for string

class ExpressionNode;
class IdentifierNode;

class AssignmentStatementNode : public StatementNode
{
public:
  AssignmentStatementNode(LvalueNode*& lval, ExpressionNode* expr);
  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

private:
  const std::shared_ptr<LvalueNode> identifier;
  const std::shared_ptr<ExpressionNode> expr;
};

#endif
