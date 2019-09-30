#ifndef STRING_CONSTANT_NODE_HPP
#define STRING_CONSTANT_NODE_HPP

#include "LiteralNode.hpp" // for LiteralNode
#include "Value.hpp"       // for Value

#include <string> // for string

class StringLiteralNode : public LiteralNode
{
public:
  StringLiteralNode(std::string);
  virtual std::variant<std::monostate, int, char, bool> eval() const override;
  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;

  const std::string string;
};

#endif
