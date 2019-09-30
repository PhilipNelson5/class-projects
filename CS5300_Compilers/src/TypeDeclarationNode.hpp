#ifndef TYPE_DECLARATION_NODE_HPP
#define TYPE_DECLARATION_NODE_HPP

#include "Node.hpp" // for Node
#include "Type.hpp"
#include "TypeNode.hpp"
#include "Value.hpp" // for Value

#include <memory> // for shared_ptr
#include <string> // for string
class Type;

class TypeDeclarationNode : public Node
{
public:
  TypeDeclarationNode(std::string ident, TypeNode*& type);
  virtual void emitSource(std::string indent) override;
  void emit();

private:
  const std::string m_id;
  const std::shared_ptr<TypeNode> m_typeNode;
};

#endif
