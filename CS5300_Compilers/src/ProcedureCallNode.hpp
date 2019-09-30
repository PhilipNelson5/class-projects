#ifndef PROCEDURE_CALL_NODE_HPP
#define PROCEDURE_CALL_NODE_HPP

#include "ListNode.hpp"      // for ListNode
#include "StatementNode.hpp" // for StatementNode

#include <memory> // for shared_ptr, allocator
#include <string> // for string
#include <vector> // for vector
class ExpressionNode;

class ProcedureCallNode : public StatementNode
{
public:
  ProcedureCallNode(char* id, ListNode<ExpressionNode>*& args)
    : id(id)
    , args(ListNode<ExpressionNode>::makeVector(args))
  {}

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

  const std::string id;
  const std::vector<std::shared_ptr<ExpressionNode>> args;
};

#endif
