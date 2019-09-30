#ifndef BOOLEAN_CONSTANT_NODE_HPP
#define BOOLEAN_CONSTANT_NODE_HPP

#include "LiteralNode.hpp"
#include "RegisterPool.hpp"

class BooleanLiteralNode : public LiteralNode
{
public:
  BooleanLiteralNode(int value);
  virtual std::variant<std::monostate, int, char, bool> eval() const override;
  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;

  const bool value;
};

#endif
