#include "StringLiteralNode.hpp"

#include "SymbolTable.hpp" // for SymbolTable, symbol_table
#include "Type.hpp"        // for StringType

#include <iostream> // for basic_ostream, cout, ostream

StringLiteralNode::StringLiteralNode(std::string string)
  : LiteralNode(StringType::get())
  , string(string)
{}

std::variant<std::monostate, int, char, bool> StringLiteralNode::eval() const
{
  return {};
}

void StringLiteralNode::emitSource(std::string indent)
{
  std::cout << indent << string;
}

Value StringLiteralNode::emit()
{
  return symbol_table.lookupString(string);
}
