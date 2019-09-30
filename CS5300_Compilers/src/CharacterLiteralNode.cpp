#include "CharacterLiteralNode.hpp"

#include "../fmt/include/fmt/core.h" // for print
#include "Type.hpp"                  // for CharacterType
#include "log/easylogging++.h"       // for Writer, CDEBUG, LOG

#include <string> // for allocator, string

CharacterLiteralNode::CharacterLiteralNode(char character)
  : LiteralNode(CharacterType::get())
  , character(character)
{
  LOG(DEBUG) << "NEW CHARACTER NODE";
}

std::string CharacterLiteralNode::toString() const
{
  switch (character)
  {
  case '\n':
    return "\\n";
  case '\r':
    return "\\r";
  case '\b':
    return "\\b";
  case '\t':
    return "\\t";
  case '\f':
    return "\\f";
  default:
    return {1, character};
  }
}

std::variant<std::monostate, int, char, bool> CharacterLiteralNode::eval() const
{
  return {character};
}

void CharacterLiteralNode::emitSource(std::string indent)
{
  (void)indent;
  fmt::print("'{}'", toString());
}

Value CharacterLiteralNode::emit()
{
  return character;
}
