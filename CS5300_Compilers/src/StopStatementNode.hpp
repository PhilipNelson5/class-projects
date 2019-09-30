#ifndef STOP_STATEMENT_NODE_HPP
#define STOP_STATEMENT_NODE_HPP

#include "StatementNode.hpp" // for StatementNode
#include "Value.hpp"         // for Value

#include <string> // for string

class StopStatementNode : public StatementNode
{
public:
  StopStatementNode();
  virtual void emitSource(std::string indent) override;
  virtual void emit() override;
};

#endif
