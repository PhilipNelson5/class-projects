#ifndef PROCEDURE_OF_FUNCTION_DECLARATION_NODE_HPP
#define PROCEDURE_OF_FUNCTION_DECLARATION_NODE_HPP

#include "Node.hpp"

class ProcedureOrFunctionDeclarationNode : public Node
{
public:
  virtual void emitSource(std::string) = 0;
  virtual void emit() = 0;
};

#endif
