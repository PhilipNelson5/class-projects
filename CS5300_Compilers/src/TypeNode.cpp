#include "Type.hpp"

#include "SymbolTable.hpp"

const std::shared_ptr<Type> TypeNode::getType()
{
  if (type == nullptr)
  {
    type = symbol_table.lookupType(id);
  }
  return type;
}

