#ifndef DECLARATION_NODE_HPP
#define DECLARATION_NODE_HPP

#include "Node.hpp" // for Node

class DeclarationNode : public Node
{
public:
  DeclarationNode() = default;
  virtual ~DeclarationNode() = default;
  virtual void emit() = 0;
};

#endif
