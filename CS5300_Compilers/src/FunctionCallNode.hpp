#ifndef FUNCTION_CALL_NODE_HPP
#define FUNCTION_CALL_NODE_HPP

#include "ExpressionNode.hpp"
#include "ListNode.hpp"
#include "Node.hpp"
#include "Value.hpp"

#include <memory>
#include <vector>

class FunctionCallNode : public ExpressionNode
{
public:
  FunctionCallNode(char* id, ListNode<ExpressionNode>*& args)
    : id(id)
    , args(ListNode<ExpressionNode>::makeVector(args))
  {}

  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;
  virtual bool isConstant() const override;
  virtual const std::shared_ptr<Type> getType() override;
  virtual std::variant<std::monostate, int, char, bool> eval() const override;

  const std::string id;
  const std::vector<std::shared_ptr<ExpressionNode>> args;
};

#endif
