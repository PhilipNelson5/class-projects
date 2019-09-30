#ifndef IDENTIFIER_NODE_HPP
#define IDENTIFIER_NODE_HPP

#include "LvalueNode.hpp" // for LvalueNode
#include "Value.hpp"      // for Value

#include <string> // for string

class IdentifierNode : public LvalueNode
{
public:
  IdentifierNode(std::string id);
  virtual bool isConstant() const override;
  virtual const std::shared_ptr<Type> getType() override;
  virtual std::string getId() const override;
  virtual std::variant<std::monostate, int, char, bool> eval() const override;
  virtual void emitSource(std::string indent) override;
  virtual Value emit() override;

  const std::string id;
};

#endif
