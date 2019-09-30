#ifndef PROCEDURE_DECLARATION_NODE_HPP
#define PROCEDURE_DECLARATION_NODE_HPP

#include "BodyNode.hpp"                           // for BodyNode
#include "ListNode.hpp"                           // for ListNode
#include "ProcedureOrFunctionDeclarationNode.hpp" // for ProcedureOrFunctio...

#include <memory> // for shared_ptr, allocator
#include <string> // for string
#include <vector> // for vector
class FormalParameter;

class ProcedureDeclarationNode : public ProcedureOrFunctionDeclarationNode
{
public:
  ProcedureDeclarationNode(char* id, ListNode<FormalParameter>*& parameters, BodyNode*& body)
    : id(id)
    , parameters(ListNode<FormalParameter>::makeVector(parameters))
    , body(body)
  {}

  virtual void emitSource(std::string indent) override;
  virtual void emit() override;

  const std::string id;
  const std::vector<std::shared_ptr<FormalParameter>> parameters;
  const std::shared_ptr<BodyNode> body;
};

#endif
