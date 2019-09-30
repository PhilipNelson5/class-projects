#ifndef FOR_STATEMENT_NODE_HPP
#define FOR_STATEMENT_NODE_HPP

#include "ExpressionNode.hpp"
#include "IdentifierNode.hpp"
#include "ListNode.hpp"
#include "StatementNode.hpp"

#include <memory>
#include <vector>

class ForStatementNode : public StatementNode
{
public:
  enum Type
  {
    TO,
    DOWNTO
  };

  ForStatementNode(IdentifierNode* identifier,
                   ExpressionNode*& startExpr,
                   ExpressionNode*& endExpr,
                   ListNode<StatementNode>*& statements,
                   Type type);

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

  const std::shared_ptr<IdentifierNode> identifier;
  const std::shared_ptr<ExpressionNode> startExpr;
  const std::shared_ptr<ExpressionNode> endExpr;
  const std::vector<std::shared_ptr<StatementNode>> statements;
  const Type type;

private:
  void printForHeader(std::string indent);
};

#endif
