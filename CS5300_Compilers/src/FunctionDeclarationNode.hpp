#ifndef FUNCTION_DECLARATION_NODE_HPP
#define FUNCTION_DECLARATION_NODE_HPP

#include "BodyNode.hpp"
#include "FormalParameter.hpp"
#include "ListNode.hpp"
#include "ProcedureOrFunctionDeclarationNode.hpp"
#include "TypeNode.hpp" // for TypeNode

#include <string> // for string
class FormalParameter;
class Type;

class FunctionDeclarationNode : public ProcedureOrFunctionDeclarationNode
{
public:
  FunctionDeclarationNode(char* id,
                          ListNode<FormalParameter>*& parameters,
                          TypeNode*& returnTypeNode,
                          BodyNode*& body)
    : id(id)
    , parameters(ListNode<FormalParameter>::makeVector(parameters))
    , returnTypeNode(returnTypeNode)
    , returnType(nullptr)
    , body(body)
  {}

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;
  std::shared_ptr<Type> getType();

  const std::string id;
  const std::vector<std::shared_ptr<FormalParameter>> parameters;
  const std::shared_ptr<TypeNode> returnTypeNode;
  std::shared_ptr<Type> returnType;
  const std::shared_ptr<BodyNode> body;
};

#endif
