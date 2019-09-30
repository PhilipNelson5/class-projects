#include "IdentifierNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "LiteralNode.hpp"           // for LiteralNode
#include "SymbolTable.hpp"           // for SymbolTable, symbol_table, Variable
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <iostream> // for basic_ostream, cout, ostream
#include <memory>   // for operator!=, shared_ptr, __shared_ptr_...
#include <stdlib.h> // for exit, EXIT_FAILURE

IdentifierNode::IdentifierNode(std::string id)
  : LvalueNode()
  , id(id)
{}

bool IdentifierNode::isConstant() const
{
  return (symbol_table.lookupConst(id) != nullptr);
}

const std::shared_ptr<Type> IdentifierNode::getType()
{
  if (type == nullptr)
  {
    type = symbol_table.getType(id);
  }
  return type;
}

std::string IdentifierNode::getId() const
{
  return id;
}

std::variant<std::monostate, int, char, bool> IdentifierNode::eval() const
{
  auto literlNode = symbol_table.lookupConst(id);
  if (literlNode == nullptr)
  {
    return {};
  }
  return literlNode->eval();
}

void IdentifierNode::emitSource(std::string indent)
{
  std::cout << indent << id;
}

Value IdentifierNode::emit()
{
  auto lval_info = symbol_table.lookupLval(id);
  if (lval_info != nullptr)
  {
    fmt::print("# {}, {}, {}\n", lval_info->identifier ,lval_info->offset, lval_info->memoryLocation);
    return {lval_info->offset, lval_info->memoryLocation};
  }

  auto const_info = symbol_table.lookupConst(id);
  if (const_info != nullptr)
  {
    return const_info->emit();
  }

  LOG(ERROR) << fmt::format("{} is not defined", id);
  exit(EXIT_FAILURE);
}

