#ifndef STATEMENT_NODE_HPP
#define STATEMENT_NODE_HPP

#include "Node.hpp" // for Node

class StatementNode : public Node
{
public:
  StatementNode() = default;
  virtual ~StatementNode() = default;
  virtual void emit() = 0;
};

#endif
