#ifndef VARIABLE_DECLARATION_NODE_HPP
#define VARIABLE_DECLARATION_NODE_HPP

#include "Node.hpp"  // for Node
#include "Value.hpp" // for Value

#include <memory> // for shared_ptr
#include <string> // for string
#include <vector> // for vector
class Type;
class TypeNode;
template<typename T>
class ListNode;

class VariableDeclarationNode : public Node
{
public:
  VariableDeclarationNode(ListNode<std::string>* identList,
                          TypeNode*& typeNode);
  virtual void emitSource(std::string indent) override;
  void emit();

  const std::vector<std::string> m_ids;
  const std::shared_ptr<TypeNode> m_typeNode;
};

#endif
