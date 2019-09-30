#ifndef EXPRESSION_NODE_HPP
#define EXPRESSION_NODE_HPP

#include "Node.hpp"  // for Node
#include "Value.hpp" // for Value

#include <memory> // for shared_ptr
#include <variant>
class Type;

class ExpressionNode : public Node
{
public:
  ExpressionNode();
  ExpressionNode(std::shared_ptr<Type> type);
  virtual ~ExpressionNode() = default;

  virtual Value emit() = 0;
  virtual const std::shared_ptr<Type> getType();
  virtual void setType(std::shared_ptr<Type> t);
  virtual bool isConstant() const = 0;
  virtual bool isLiteral() const;
  virtual std::variant<std::monostate, int, char, bool> eval() const = 0;

protected:
  std::shared_ptr<Type> type;
};

#endif
