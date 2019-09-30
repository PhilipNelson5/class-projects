#ifndef CONST_DECL_NODE_HPP
#define CONST_DECL_NODE_HPP

#include "DeclarationNode.hpp"
#include "Value.hpp" // for Value

#include <memory> // for shared_ptr
#include <string> // for string

class ExpressionNode;
class LiteralNode;

class ConstantDeclarationNode : public DeclarationNode
{
public:
  ConstantDeclarationNode(std::string ident, ExpressionNode* type);
  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

private:
  const std::string m_id;
  const std::shared_ptr<LiteralNode> m_expr;
};

#endif
