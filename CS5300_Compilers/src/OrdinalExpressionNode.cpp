#include "OrdinalExpressionNode.hpp"

#include "../fmt/include/fmt/core.h" // for format
#include "Type.hpp"                  // for IntegerType, CharacterType, Type
#include "Value.hpp"                 // for Value

#include <iostream> // for operator<<, cout, ostream, basi...
#include <stdlib.h> // for exit, EXIT_FAILURE

OrdinalExpressionNode::OrdinalExpressionNode(ExpressionNode*& expr)
  : ExpressionNode(IntegerType::get())
  , expr(expr)
{}

bool OrdinalExpressionNode::isConstant() const
{
  return expr->isConstant();
}

std::variant<std::monostate, int, char, bool> OrdinalExpressionNode::eval() const
{
  auto var_expr = expr->eval();

  if (var_expr.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<char>(var_expr))
  {
    return static_cast<int>(std::get<char>(var_expr));
  }

  fmt::print("ord is not defined on {}. Must use character type", expr->getType()->name());
  exit(EXIT_FAILURE);
}

void OrdinalExpressionNode::emitSource(std::string indent)
{
  std::cout << indent << "ord(";
  expr->emitSource("");
  std::cout << ")";
}

Value OrdinalExpressionNode::emit()
{
  if (expr->getType() != CharacterType::get())
  {
    fmt::print("ord is not defined on {}. Must use character type", expr->getType()->name());
    exit(EXIT_FAILURE);
  }

  expr->setType(IntegerType::get());

  return expr->emit();
}
