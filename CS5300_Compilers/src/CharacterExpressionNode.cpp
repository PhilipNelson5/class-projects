#include "CharacterExpressionNode.hpp"

#include "../fmt/include/fmt/core.h" // for format
#include "Type.hpp"                  // for CharacterType, IntegerType, Type
#include "log/easylogging++.h"       // for Writer, CERROR, LOG

#include <iostream> // for operator<<, char_traits, cout
#include <stdlib.h> // for exit, EXIT_FAILURE

CharacterExpressionNode::CharacterExpressionNode(ExpressionNode*& expr)
  : ExpressionNode(CharacterType::get())
  , expr(expr)
{}

bool CharacterExpressionNode::isConstant() const
{
  return expr->isConstant();
}

std::variant<std::monostate, int, char, bool> CharacterExpressionNode::eval() const
{
  auto var_expr = expr->eval();

  if (var_expr.index() == 0)
  {
    return {};
  }
  if (std::holds_alternative<int>(var_expr))
  {
    return static_cast<char>(std::get<int>(var_expr));
  }

  LOG(ERROR) << fmt::format("chr is not defined on {}. Must use integer type",
                            expr->getType()->name());
  exit(EXIT_FAILURE);
}

void CharacterExpressionNode::emitSource(std::string indent)
{
  std::cout << indent << "chr(";
  expr->emitSource("");
  std::cout << ")";
}

Value CharacterExpressionNode::emit()
{
  if (expr->getType() != IntegerType::get())
  {
    LOG(ERROR) << fmt::format("chr is not defined on {}. Must use integer type",
                              expr->getType()->name());
    exit(EXIT_FAILURE);
  }

  expr->setType(CharacterType::get());

  return expr->emit();
}
