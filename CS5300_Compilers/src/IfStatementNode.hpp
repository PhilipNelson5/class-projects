#ifndef IF_STATEMENT_NODE_HPP
#define IF_STATEMENT_NODE_HPP

#include "StatementNode.hpp" // for StatementNode
#include "Value.hpp"         // for Value

#include <memory>     // for shared_ptr
#include <string>     // for string
#include <utility>    // for pair
#include <vector>     // for vector
class ExpressionNode;
template<typename T>
class ListNode;

using IfStatement
  = std::pair<std::shared_ptr<ExpressionNode>, std::vector<std::shared_ptr<StatementNode>>>;

class IfStatementNode : public StatementNode
{
public:
  IfStatementNode(ExpressionNode*& ifExpr,
                  ListNode<StatementNode>*& ifStatements,
                  ListNode<IfStatement>*& elseIfStatements,
                  ListNode<StatementNode>*& elseStatement);
  virtual void emitSource(std::string indent) override;
  virtual ~IfStatementNode() override = default;
  virtual void emit() override;

  const IfStatement ifStatement;
  const std::vector<IfStatement> elseIfStatements;
  const std::vector<std::shared_ptr<StatementNode>> elseStatement;
};

#endif
